from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Protocol

from config import Settings


logger = logging.getLogger(__name__)


class OptionalRetriever(Protocol):
    """Interface placeholder for future modality retrievers."""

    def search(self, query: str, top_k: int) -> List[Dict]:
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

        # Phase 3: Confidence-based routing replaces skip_non_event.
        # All modalities always run; fusion strategy is determined by route_info.
        route_info = self.route_by_confidence(intent)

        # CLIP always runs (optimization 1: no more skip_non_event)
        clip_results = self.clip_search_service.search_text_variants(
            rewritten_queries,
            top_k=candidate_top_k,
        )
        detection_results = []
        if self.detection_retriever is not None:
            detection_results = self.detection_retriever.search_as_frames(
                query=query,
                top_k=candidate_top_k,
            )
        trajectory_results = []
        if self.trajectory_retriever is not None:
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

        ocr_results = []
        if self.ocr_search_service is not None:
            ocr_results = self.ocr_search_service.search_as_frames(
                query=query,
                top_k=candidate_top_k,
            )

        fused_results = self._fuse_results(
            clip_results=clip_results,
            detection_results=detection_results,
            trajectory_results=trajectory_results,
            event_results=event_results,
            ocr_results=ocr_results,
            intent=intent,
            route_info=route_info,
        )
        aggregated_results = self.result_aggregation_service.aggregate_results(
            fused_results,
            top_k=top_k,
            score_threshold=self.settings.hybrid_score_threshold,
        )
        event_conf = float(self._intent_value(intent, "event_confidence") or 0.0)
        logger.info(
            "Hybrid search completed query=%s kind=%s event_conf=%.2f route=%s variants=%s clip_results=%s detection_results=%s trajectory_results=%s event_results=%s ocr_results=%s aggregated=%s",
            query,
            intent["kind"],
            event_conf,
            route_info["route"],
            len(rewritten_queries),
            len(clip_results),
            len(detection_results),
            len(trajectory_results),
            len(event_results),
            len(ocr_results),
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
        ocr_results: List[Dict],
        intent: Dict[str, object],
        route_info: Dict[str, Any] | None = None,
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

        for item in ocr_results:
            key = self._result_key(item)
            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    **item,
                    "best_score": float(item.get("score", 0.0)),
                    "ocr_score": float(item.get("ocr_score", item.get("score", 0.0))),
                    "start_ts": float(item.get("start_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "end_ts": float(item.get("end_ts", item.get("timestamp_seconds", item.get("timestamp", 0.0)))),
                    "timestamp_seconds": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                }
                continue

            self._merge_temporal_bounds(existing, item)
            self._merge_evidence(existing, item, "ocr_score")
            existing["ocr_score"] = max(
                float(existing.get("ocr_score", 0.0)),
                float(item.get("ocr_score", item.get("score", 0.0))),
            )
            if item.get("matched_text") and not existing.get("matched_text"):
                existing["matched_text"] = item["matched_text"]

        return [self._finalize_fused_item(item, intent, route_info) for item in fused.values()]

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

    def _finalize_fused_item(
        self,
        item: Dict,
        intent: Dict[str, object],
        route_info: Dict[str, Any] | None = None,
    ) -> Dict:
        """Fuse multimodal scores using routing-aware strategy.

        Phase 3 optimization: Uses route_by_confidence to determine fusion:
        - event_primary: CLIP base + event boost(0.12)
        - hybrid_balanced: weighted average across modalities
        - clip_primary: CLIP base + event boost(0.04)
        - vague_event: CLIP base + event boost(0.06)
        - clip_only: CLIP dominant weighted average
        """
        if route_info is None:
            route_info = {"route": "clip_only", "fusion": "weighted_average"}
        route = route_info.get("route", "clip_only")

        clip_score = self._normalize_clip_score(float(item.get("clip_score", 0.0)))
        detection_score = min(float(item.get("detection_score", 0.0)), 1.0)
        trajectory_score = min(float(item.get("trajectory_score", 0.0)), 1.0)
        event_score = min(float(item.get("event_score", 0.0)), 1.0)
        ocr_score = min(float(item.get("ocr_score", 0.0)), 1.0)

        if route in ("event_primary", "clip_primary", "vague_event"):
            # CLIP-primary base + event dynamic boost
            final_score = clip_score

            if event_score > 0:
                event_boost = route_info.get("event_boost", 0.02)
                final_score += event_boost * event_score
            if detection_score > 0:
                final_score += 0.03 * detection_score
            if trajectory_score > 0:
                final_score += 0.03 * trajectory_score
            if ocr_score > 0:
                final_score += 0.02 * ocr_score
        else:
            # clip_only: CLIP dominant weighted average
            weights = route_info.get("weights", {
                "clip": 0.50, "detection": 0.25, "ocr": 0.15, "trajectory": 0.10,
            })
            final_score = (
                weights["clip"] * clip_score
                + weights["detection"] * detection_score
                + weights["ocr"] * ocr_score
                + weights["trajectory"] * trajectory_score
            )

        # Cap at 1.0
        final_score = min(final_score, 1.0)

        # Consensus bonus: reward items matched by multiple modalities
        active_modalities = sum(1 for s in [clip_score, detection_score, trajectory_score, event_score, ocr_score] if s > 0)
        consensus_bonus = max(active_modalities - 1, 0) * 0.05
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
            return {"clip": 0.10, "detection": 0.10, "trajectory": 0.15, "event": 0.60, "ocr": 0.05}
        if kind == "motion":
            return {"clip": 0.10, "detection": 0.20, "trajectory": 0.55, "event": 0.10, "ocr": 0.05}
        if kind == "relational":
            return {"clip": 0.10, "detection": 0.50, "trajectory": 0.20, "event": 0.10, "ocr": 0.10}
        if kind == "composite":
            return {"clip": 0.05, "detection": 0.30, "trajectory": 0.30, "event": 0.25, "ocr": 0.10}
        if kind == "object":
            return {"clip": 0.15, "detection": 0.45, "trajectory": 0.15, "event": 0.10, "ocr": 0.15}
        if kind == "attribute":
            return {"clip": 0.15, "detection": 0.40, "trajectory": 0.15, "event": 0.15, "ocr": 0.15}
        if kind == "semantic":
            return {"clip": 0.65, "detection": 0.15, "trajectory": 0.10, "event": 0.05, "ocr": 0.05}
        return {"clip": 0.35, "detection": 0.25, "trajectory": 0.20, "event": 0.10, "ocr": 0.10}

    def generate_dynamic_weights(self, intent) -> Dict[str, float]:
        """Generate dynamic channel weights based on intent parsing results.

        Layer 4 of TQUM: Routes retrieval channels based on event confidence
        and attribute richness instead of static query_type mapping.

        Routing logic:
        - event_confidence >= 0.8, no visual attr → event channel dominant (0.65)
        - event_confidence >= 0.8, has visual attr → CLIP + event balanced (0.35/0.40)
        - event_confidence 0.6-0.8 → event weighted (0.35) + detection support
        - attribute + entity query → CLIP dominant (0.50)
        - vague query → balanced multi-channel

        Color attribute boosts CLIP weight (CLIP excels at color matching).
        """
        event_conf = float(self._intent_value(intent, "event_confidence") or 0.0)
        attributes = self._intent_value(intent, "attributes") or {}
        primary_entities = self._intent_value(intent, "primary_entities") or []

        has_attributes = bool(attributes)
        has_entities = bool(primary_entities)
        has_visual_attr = bool(
            attributes.get("color") or attributes.get("vehicle_type")
        )

        if event_conf >= 0.8 and not has_visual_attr:
            # High confidence event, no visual attributes → event dominant
            base = {"clip": 0.05, "detection": 0.10, "trajectory": 0.15,
                    "event": 0.65, "ocr": 0.05}
        elif event_conf >= 0.8 and has_visual_attr:
            # High confidence event + visual attributes → balanced CLIP + event
            # CLIP needs enough weight to differentiate between videos with same event type
            base = {"clip": 0.35, "detection": 0.10, "trajectory": 0.10,
                    "event": 0.35, "ocr": 0.10}
        elif event_conf >= 0.6:
            # Medium confidence event → event + detection dual path
            base = {"clip": 0.10, "detection": 0.25, "trajectory": 0.20,
                    "event": 0.35, "ocr": 0.10}
        elif has_attributes and has_entities:
            # Attribute + entity → CLIP dominant
            base = {"clip": 0.50, "detection": 0.25, "trajectory": 0.10,
                    "event": 0.05, "ocr": 0.10}
        else:
            # Vague query → balanced
            base = {"clip": 0.30, "detection": 0.20, "trajectory": 0.15,
                    "event": 0.25, "ocr": 0.10}

        # Color attribute boosts CLIP (CLIP is good at color matching)
        if attributes.get("color"):
            base["clip"] = min(base["clip"] + 0.10, 0.60)
            total = sum(base.values())
            base = {k: v / total for k, v in base.items()}

        logger.debug(
            "Dynamic weights event_conf=%.2f has_attr=%s has_visual=%s has_entities=%s weights=%s",
            event_conf, has_attributes, has_visual_attr, has_entities, base,
        )
        return base

    def route_by_confidence(self, intent) -> Dict[str, Any]:
        """Route retrieval based on confidence levels.

        Phase 4 optimization: Event-type-specific boost values.

        Rare events (red_light_violation) get a higher boost to surface
        the few videos that contain them. Common events (wrong_way_driving)
        keep a low boost since CLIP already ranks them well.

        Boost values:
        - red_light_violation: 0.08 (rare, only 1-2 videos)
        - vehicle_crosses_line: 0.05 (common but discriminative)
        - wrong_way_driving: 0.03 (very common, CLIP handles well)
        - vague: 0.03 (all events, low priority)

        Routing logic:
        - event_conf >= 0.8 → event_primary (type-specific boost)
        - event_conf 0-0.8 → clip_primary (type-specific boost * 0.7)
        - no event + vague keywords → vague_event (boost 0.03, all events)
        - otherwise → clip_only
        """
        event_conf = float(self._intent_value(intent, "event_confidence") or 0.0)
        event_types = self._intent_value(intent, "event_types") or []
        has_event = bool(event_types)

        # Check for vague violation intent from intent_candidates
        intent_candidates = self._intent_value(intent, "intent_candidates") or []
        has_vague = any(c.get("vague") for c in intent_candidates)

        # Event-type-specific boost: rare events need stronger signal
        EVENT_BOOST_MAP = {
            "red_light_violation": 0.15,
            "vehicle_crosses_line": 0.05,
            "wrong_way_driving": 0.03,
        }
        primary_event_boost = max(
            (EVENT_BOOST_MAP.get(et, 0.03) for et in event_types),
            default=0.03,
        )

        if has_event and event_conf >= 0.8:
            return {
                "route": "event_primary",
                "fusion": "clip_base + event_boost",
                "event_boost": primary_event_boost,
                "skip_clip": False,
            }
        elif has_event and event_conf > 0:
            return {
                "route": "clip_primary",
                "fusion": "clip_base + event_boost",
                "event_boost": primary_event_boost * 0.7,
                "skip_clip": False,
            }
        elif has_vague:
            return {
                "route": "vague_event",
                "fusion": "clip_base + event_boost",
                "event_boost": 0.03,
                "skip_clip": False,
            }
        else:
            return {
                "route": "clip_only",
                "fusion": "weighted_average",
                "weights": {
                    "clip": 0.50, "detection": 0.25, "ocr": 0.15, "trajectory": 0.10,
                },
            }

    def _normalize_clip_score(self, score: float) -> float:
        return min(max(score, 0.0) / 0.35, 1.0)

    def _intent_value(self, intent, key: str):
        if hasattr(intent, key):
            return getattr(intent, key)
        if isinstance(intent, dict):
            return intent.get(key)
        return None
