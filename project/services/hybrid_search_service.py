from __future__ import annotations

import logging
import math
from typing import Dict, List, Protocol

from config import Settings


logger = logging.getLogger(__name__)


class OptionalRetriever(Protocol):
    """Interface placeholder for future modality retrievers."""

    def search(self, query_text: str, top_k: int) -> List[Dict]:
        """Return frame- or segment-like search results."""

    def search_as_frames(self, query: str, top_k: int) -> List[Dict]:
        """Return frame- or segment-like results adapted for hybrid fusion."""


class HybridSearchService:
    """Coordinate query rewriting, multimodal retrieval, fusion, and aggregation."""

    def __init__(
        self,
        settings: Settings,
        query_rewrite_service,
        clip_search_service,
        ocr_search_service,
        result_aggregation_service,
        asr_retriever: OptionalRetriever | None = None,
        detection_retriever: OptionalRetriever | None = None,
        trajectory_retriever: OptionalRetriever | None = None,
        event_retriever: OptionalRetriever | None = None,
    ) -> None:
        self.settings = settings
        self.query_rewrite_service = query_rewrite_service
        self.clip_search_service = clip_search_service
        self.ocr_search_service = ocr_search_service
        self.result_aggregation_service = result_aggregation_service
        self.asr_retriever = asr_retriever
        self.detection_retriever = detection_retriever
        self.trajectory_retriever = trajectory_retriever
        self.event_retriever = event_retriever

    def search(self, query: str, top_k: int = 20) -> Dict[str, object]:
        """Run hybrid retrieval and return segment-level results."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        intent = self.query_rewrite_service.parse_query_intent(query)
        rewritten_queries = intent["rewritten_queries"]
        candidate_top_k = max(top_k, top_k * self.settings.hybrid_candidate_multiplier)
        intent_query_type = self._intent_value(intent, "query_type") or self._intent_value(intent, "kind") or "general"

        clip_results = []
        if intent_query_type != "event":
            clip_results = self.clip_search_service.search_text_variants(
                rewritten_queries,
                top_k=candidate_top_k,
            )
        detection_results = []
        if self.detection_retriever is not None and intent_query_type != "event":
            detection_results = self.detection_retriever.search_as_frames(
                query=query,
                top_k=candidate_top_k,
            )
        trajectory_results = []
        if self.trajectory_retriever is not None and intent_query_type != "event":
            trajectory_results = self.trajectory_retriever.search_as_frames(
                query=query,
                top_k=candidate_top_k,
            )
        event_results = []
        if self.event_retriever is not None:
            event_results = self.event_retriever.search_as_frames(
                query=query,
                top_k=candidate_top_k,
            )

        fused_results = self._fuse_results(
            clip_results=clip_results,
            detection_results=detection_results,
            trajectory_results=trajectory_results,
            event_results=event_results,
            intent=intent,
        )
        aggregated_results = self.result_aggregation_service.aggregate_results(
            fused_results,
            top_k=top_k,
            score_threshold=self.settings.hybrid_score_threshold,
        )
        logger.info(
            "Hybrid search completed query=%s kind=%s variants=%s clip_results=%s detection_results=%s trajectory_results=%s event_results=%s aggregated=%s",
            query,
            intent["kind"],
            len(rewritten_queries),
            len(clip_results),
            len(detection_results),
            len(trajectory_results),
            len(event_results),
            len(aggregated_results),
        )
        return {
            "query": query,
            "rewritten_queries": rewritten_queries,
            "results": aggregated_results,
        }

    def _fuse_results(
        self,
        clip_results: List[Dict],
        detection_results: List[Dict],
        trajectory_results: List[Dict],
        event_results: List[Dict],
        intent: Dict[str, object],
    ) -> List[Dict]:
        """Fuse modality results by video-time windows and intent-aware scoring."""
        fused: Dict[str, Dict] = {}

        for item in clip_results:
            key = self._result_key(item)
            fused[key] = {
                **item,
                "matched_by": ["clip"],
                "best_score": float(item.get("score", 0.0)),
                "clip_score": float(item.get("score", 0.0)),
                "start_ts": float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                "end_ts": float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                "timestamp_seconds": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
            }

        for item in detection_results:
            key = self._result_key(item)
            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    **item,
                    "best_score": float(item.get("score", 0.0)),
                    "detection_score": float(item.get("detection_score", item.get("score", 0.0))),
                    "start_ts": float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "end_ts": float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "timestamp_seconds": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                }
                continue

            self._merge_temporal_bounds(existing, item)
            self._merge_evidence(existing, item, "detection_score")
            existing["detection_score"] = max(
                float(existing.get("detection_score", 0.0)),
                float(item.get("detection_score", item.get("score", 0.0))),
            )
            if item.get("matched_label") and not existing.get("matched_label"):
                existing["matched_label"] = item["matched_label"]

        for item in trajectory_results:
            key = self._result_key(item)
            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    **item,
                    "best_score": float(item.get("score", 0.0)),
                    "trajectory_score": float(item.get("trajectory_score", item.get("score", 0.0))),
                    "start_ts": float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "end_ts": float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "timestamp_seconds": float(item.get("timestamp_seconds", item.get("start_ts", 0.0))),
                }
                continue

            self._merge_temporal_bounds(existing, item)
            self._merge_evidence(existing, item, "trajectory_score")
            existing["trajectory_score"] = max(
                float(existing.get("trajectory_score", 0.0)),
                float(item.get("trajectory_score", item.get("score", 0.0))),
            )
            for field_name in ("matched_label", "matched_direction", "track_id"):
                if item.get(field_name) and not existing.get(field_name):
                    existing[field_name] = item[field_name]

        for item in event_results:
            key = self._result_key(item)
            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    **item,
                    "best_score": float(item.get("score", 0.0)),
                    "event_score": float(item.get("event_score", item.get("score", 0.0))),
                    "start_ts": float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "end_ts": float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "timestamp_seconds": float(item.get("timestamp_seconds", item.get("start_ts", 0.0))),
                }
                continue

            self._merge_temporal_bounds(existing, item)
            self._merge_evidence(existing, item, "event_score")
            existing["event_score"] = max(
                float(existing.get("event_score", 0.0)),
                float(item.get("event_score", item.get("score", 0.0))),
            )
            for field_name in ("matched_event_type", "event_id", "matched_label", "track_id"):
                if item.get(field_name) and not existing.get(field_name):
                    existing[field_name] = item[field_name]

        return [self._finalize_fused_item(item, intent) for item in fused.values()]

    def _result_key(self, item: Dict) -> str:
        """Build a stable fusion key for multimodal results."""
        video_id = str(item.get("video_id", ""))
        if not video_id:
            return str(item.get("frame_path") or item.get("frame_id") or "")

        anchor = self._anchor_timestamp(item)
        bucket = int(math.floor(anchor / self.settings.segment_window_seconds))
        return f"{video_id}:{bucket}"

    def _anchor_timestamp(self, item: Dict) -> float:
        if "timestamp_seconds" in item:
            return float(item["timestamp_seconds"])
        if "start_ts" in item and "end_ts" in item:
            return (float(item["start_ts"]) + float(item["end_ts"])) / 2.0
        if "start_ts" in item:
            return float(item["start_ts"])
        return float(item.get("timestamp", 0.0))

    def _merge_temporal_bounds(self, existing: Dict, item: Dict) -> None:
        item_start = float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0))))
        item_end = float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0))))
        existing["start_ts"] = min(float(existing.get("start_ts", item_start)), item_start)
        existing["end_ts"] = max(float(existing.get("end_ts", item_end)), item_end)
        existing["timestamp_seconds"] = min(
            float(existing.get("timestamp_seconds", item_start)),
            float(item.get("timestamp_seconds", item_start)),
        )

    def _merge_evidence(self, existing: Dict, item: Dict, score_field: str) -> None:
        existing["best_score"] = max(
            float(existing.get("best_score", existing.get("score", 0.0))),
            float(item.get("score", 0.0)),
        )
        existing["score"] = existing["best_score"]
        existing["matched_by"] = sorted(set(existing.get("matched_by", [])) | set(item.get("matched_by", [])))
        if item.get("thumbnail_frame") and not existing.get("thumbnail_frame"):
            existing["thumbnail_frame"] = item["thumbnail_frame"]
        if item.get("frame_path") and not existing.get("frame_path"):
            existing["frame_path"] = item["frame_path"]
        if score_field not in existing:
            existing[score_field] = float(item.get(score_field, item.get("score", 0.0)))

    def _finalize_fused_item(self, item: Dict, intent: Dict[str, object]) -> Dict:
        weights = self._weights_for_intent(str(intent.get("kind", "general")))
        clip_score = self._normalize_clip_score(float(item.get("clip_score", 0.0)))
        detection_score = min(float(item.get("detection_score", 0.0)), 1.0)
        trajectory_score = min(float(item.get("trajectory_score", 0.0)), 1.0)
        event_score = min(float(item.get("event_score", 0.0)), 1.0)

        components = []
        if clip_score > 0:
            components.append(("clip", weights["clip"], clip_score))
        if detection_score > 0:
            components.append(("detection", weights["detection"], detection_score))
        if trajectory_score > 0:
            components.append(("trajectory", weights["trajectory"], trajectory_score))
        if event_score > 0:
            components.append(("event", weights["event"], event_score))

        if components:
            weighted_sum = sum(weight * value for _, weight, value in components)
            total_weight = sum(weight for _, weight, _ in components)
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            final_score = float(item.get("best_score", item.get("score", 0.0)))

        consensus_bonus = max(len({name for name, _, _ in components}) - 1, 0) * 0.05
        if intent.get("label_candidates") and item.get("matched_label") in intent.get("label_candidates"):
            consensus_bonus += 0.03
        if intent.get("direction") and item.get("matched_direction") == intent.get("direction"):
            consensus_bonus += 0.03
        if intent.get("attributes", {}).get("light_state") and item.get("matched_event_type"):
            consensus_bonus += 0.04

        item["score"] = min(final_score + consensus_bonus, 1.0)
        item["best_score"] = item["score"]
        return item

    def _weights_for_intent(self, kind: str) -> Dict[str, float]:
        if kind == "event":
            return {"clip": 0.05, "detection": 0.10, "trajectory": 0.15, "event": 0.70}
        if kind == "motion":
            return {"clip": 0.10, "detection": 0.20, "trajectory": 0.55, "event": 0.15}
        if kind == "relational":
            return {"clip": 0.10, "detection": 0.55, "trajectory": 0.20, "event": 0.15}
        if kind == "composite":
            return {"clip": 0.05, "detection": 0.30, "trajectory": 0.35, "event": 0.30}
        if kind == "object":
            return {"clip": 0.15, "detection": 0.50, "trajectory": 0.20, "event": 0.15}
        if kind == "semantic":
            return {"clip": 0.70, "detection": 0.15, "trajectory": 0.10, "event": 0.05}
        return {"clip": 0.40, "detection": 0.25, "trajectory": 0.20, "event": 0.15}

    def _normalize_clip_score(self, score: float) -> float:
        return min(max(score, 0.0) / 0.35, 1.0)

    def _intent_value(self, intent, key: str):
        if hasattr(intent, key):
            return getattr(intent, key)
        if isinstance(intent, dict):
            return intent.get(key)
        return None
