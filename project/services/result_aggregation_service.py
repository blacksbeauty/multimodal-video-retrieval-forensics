from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence

from config import Settings


logger = logging.getLogger(__name__)


class ResultAggregationService:
    """Aggregate frame-level retrieval outputs into de-duplicated segment results."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def aggregate_results(
        self,
        results: Sequence[Dict],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[Dict]:
        """Group results by video and time window, keeping the best-scoring frame per segment."""
        threshold = self.settings.hybrid_score_threshold if score_threshold is None else score_threshold
        grouped: Dict[tuple[str, int], Dict] = {}

        for item in results:
            score = float(item.get("score", item.get("best_score", 0.0)))
            if score < threshold:
                continue

            normalized = self._normalize_result(item, score)
            window_index = int(math.floor(normalized["timestamp_seconds"] / self.settings.segment_window_seconds))
            group_key = (normalized["video_id"], window_index)

            existing = grouped.get(group_key)
            if existing is None or normalized["best_score"] > existing["best_score"]:
                grouped[group_key] = normalized
            elif normalized["best_score"] == existing["best_score"]:
                existing["matched_by"] = sorted(set(existing["matched_by"]) | set(normalized["matched_by"]))

        aggregated = sorted(
            grouped.values(),
            key=lambda item: (-item["best_score"], item["video_name"], item["start_ts"]),
        )
        logger.info(
            "Aggregated retrieval results input=%s output=%s threshold=%s",
            len(results),
            len(aggregated),
            threshold,
        )
        return aggregated[:top_k]

    def _normalize_result(self, item: Dict, score: float) -> Dict:
        timestamp_seconds = float(item.get("timestamp_seconds", item.get("timestamp", item.get("start_ts", 0.0))))
        window_start = (
            math.floor(timestamp_seconds / self.settings.segment_window_seconds)
            * self.settings.segment_window_seconds
        )
        window_end = window_start + self.settings.segment_window_seconds

        matched_by = item.get("matched_by", ["clip"])
        if isinstance(matched_by, str):
            matched_by = [matched_by]

        return {
            "video_id": str(item.get("video_id", "")),
            "video_name": str(item.get("video_name", "")),
            "video_path": str(item.get("video_path", "")),
            "start_ts": float(item.get("start_ts", window_start)),
            "end_ts": float(item.get("end_ts", max(window_end, timestamp_seconds))),
            "best_score": score,
            "matched_by": sorted({str(value) for value in matched_by}),
            "clip_score": float(item.get("clip_score", 0.0)),
            "detection_score": float(item.get("detection_score", 0.0)),
            "trajectory_score": float(item.get("trajectory_score", 0.0)),
            "event_score": float(item.get("event_score", 0.0)),
            "matched_label": str(item.get("matched_label", "")),
            "matched_direction": str(item.get("matched_direction", "")),
            "matched_event_type": str(item.get("matched_event_type", "")),
            "track_id": str(item.get("track_id", "")),
            "event_id": str(item.get("event_id", "")),
            "thumbnail_frame": str(item.get("thumbnail_frame", item.get("frame_path", ""))),
            "frame_id": str(item.get("frame_id", "")),
            "timestamp_seconds": timestamp_seconds,
        }
