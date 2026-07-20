from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from config import Settings
from core.schemas import TrajectorySearchResult, TrajectoryVideoMetadata


logger = logging.getLogger(__name__)

KNOWN_DIRECTIONS = {
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
    "stationary",
    "unknown",
}


class TrajectorySearchService:
    """Search trajectory metadata using label and optional direction filters."""

    def __init__(self, settings: Settings, query_rewrite_service) -> None:
        self.settings = settings
        self.query_rewrite_service = query_rewrite_service
        self.base_dir = self.settings.trajectory_metadata_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def search_tracks(
        self,
        query: str,
        top_k: int = 10,
        direction: str | None = None,
        min_duration_sec: float | None = None,
    ) -> List[TrajectorySearchResult]:
        """Search trajectory metadata by normalized object label and filters."""
        parsed = self.query_rewrite_service.parse_query_intent(query)
        label_candidates = [self._normalize_entity_name(label) for label in parsed.get("label_candidates", []) if label]
        if parsed.query_type == "event":
            return []
        direction_filter = self.normalize_direction(direction or parsed["direction"])

        if not label_candidates:
            raise ValueError("Trajectory search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        results: List[TrajectorySearchResult] = []
        metadata_files = self.list_metadata()
        if not metadata_files:
            logger.warning("No trajectory metadata files available for trajectory search.")
            return []

        for metadata_path in metadata_files:
            metadata = self.load_metadata(metadata_path)
            if metadata is None:
                continue

            for track in metadata.tracks:
                if self.normalize_label(track.label) not in label_candidates:
                    continue
                if direction_filter and self.normalize_direction(track.direction) != direction_filter:
                    continue
                if min_duration_sec is not None and track.duration_sec < min_duration_sec:
                    continue

                results.append(
                    TrajectorySearchResult(
                        video_name=metadata.video_name,
                        video_path=metadata.video_path,
                        track_id=track.track_id,
                        label=track.label,
                        start_ts=track.start_ts,
                        end_ts=track.end_ts,
                        duration_sec=track.duration_sec,
                        direction=track.direction,
                        avg_confidence=track.avg_confidence,
                        representative_frame=track.representative_frame,
                    )
                )

        ordered = sorted(
            results,
            key=lambda item: (-item.avg_confidence, -item.duration_sec, item.video_name, item.start_ts),
        )
        logger.info(
            "Completed trajectory search labels=%s direction=%s results=%s",
            label_candidates,
            direction_filter,
            len(ordered),
        )
        return ordered[:top_k]

    def search_as_frames(
        self,
        query: str,
        top_k: int = 10,
        direction: str | None = None,
        min_duration_sec: float | None = None,
    ) -> List[Dict]:
        """Search trajectory metadata and adapt results into hybrid segment-like records."""
        raw_results = self.search_tracks(
            query=query,
            top_k=top_k,
            direction=direction,
            min_duration_sec=min_duration_sec,
        )
        return [
            {
                "video_id": self._extract_video_id(item.track_id),
                "video_name": item.video_name,
                "video_path": item.video_path,
                "frame_id": f"trajectory::{item.track_id}",
                "frame_path": item.representative_frame,
                "thumbnail_frame": item.representative_frame,
                "start_ts": item.start_ts,
                "end_ts": item.end_ts,
                "timestamp_seconds": item.start_ts,
                "score": float(item.avg_confidence),
                "trajectory_score": float(item.avg_confidence),
                "matched_by": ["trajectory"],
                "matched_label": item.label,
                "matched_direction": item.direction,
                "track_id": item.track_id,
            }
            for item in raw_results
        ]

    def list_metadata(self) -> List[Path]:
        """List all persisted trajectory metadata files."""
        return sorted(self.base_dir.glob("*.json"))

    def load_metadata(self, metadata_path: str | Path) -> TrajectoryVideoMetadata | None:
        """Load one trajectory metadata file and validate its payload."""
        path = Path(metadata_path)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return TrajectoryVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load trajectory metadata: %s", path)
            return None

    def normalize_label(self, label: str | None) -> str:
        """Normalize labels and queries for stable comparison."""
        if not label:
            return ""
        return re.sub(r"\s+", " ", label.strip().casefold())

    def normalize_direction(self, direction: str | None) -> str:
        """Normalize and validate a direction token."""
        if not direction:
            return ""
        normalized = direction.strip().casefold().replace("-", "_").replace(" ", "_")
        return normalized if normalized in KNOWN_DIRECTIONS else ""

    def _normalize_entity_name(self, entity: str) -> str:
        return self.normalize_label(entity.replace("_", " "))

    def _extract_video_id(self, track_id: str) -> str:
        """Recover the video identifier portion from a compound trajectory track id."""
        return track_id.split(":", 1)[0] if ":" in track_id else ""
