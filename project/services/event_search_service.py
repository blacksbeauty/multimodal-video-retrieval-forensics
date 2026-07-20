from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from config import Settings
from core.schemas import EventSearchResult, EventVideoMetadata


logger = logging.getLogger(__name__)

EVENT_TYPE_ALIASES = {
    "vehicle_crosses_line": ("vehicle_crosses_line", "cross line", "line crossing"),
    "red_light_violation": ("red_light_violation", "red light violation", "red light"),
}


class EventSearchService:
    """Search generated event metadata using deterministic event type matching."""

    def __init__(self, settings: Settings, query_rewrite_service=None) -> None:
        self.settings = settings
        self.base_dir = self.settings.event_metadata_dir
        self.query_rewrite_service = query_rewrite_service
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, top_k: int = 10) -> List[EventSearchResult]:
        """Search event metadata by normalized event type."""
        normalized_event_types = self._expand_event_types(query)
        if not normalized_event_types:
            raise ValueError("Event search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        results: List[EventSearchResult] = []
        for metadata_path in self.list_metadata():
            metadata = self.load_metadata(metadata_path)
            if metadata is None:
                continue

            for event in metadata.events:
                if self._normalize_event_type(event.event_type) not in normalized_event_types:
                    continue

                results.append(
                    EventSearchResult(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        plugin_name=event.plugin_name,
                        video_name=metadata.video_name,
                        video_path=metadata.video_path,
                        start_ts=event.start_ts,
                        end_ts=event.end_ts,
                        track_ids=event.track_ids,
                        confidence=event.confidence,
                        representative_frame=event.representative_frame,
                        description=event.description,
                        attributes=event.attributes,
                    )
                )

        ordered = sorted(results, key=lambda item: (-item.confidence, item.video_name, item.start_ts, item.event_type))
        logger.info("Completed event search event_types=%s results=%s", normalized_event_types, len(ordered))
        return ordered[:top_k]

    def search_as_frames(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search events and adapt them into hybrid segment-like records."""
        raw_results = self.search(query=query, top_k=top_k)
        return [
            {
                "video_id": self._extract_video_id(result.event_id),
                "video_name": result.video_name,
                "video_path": result.video_path,
                "frame_id": f"event::{result.event_id}",
                "frame_path": result.representative_frame,
                "thumbnail_frame": result.representative_frame,
                "start_ts": result.start_ts,
                "end_ts": result.end_ts,
                "timestamp_seconds": result.start_ts,
                "score": float(result.confidence),
                "event_score": float(result.confidence),
                "matched_by": ["event"],
                "matched_event_type": result.event_type,
                "matched_label": str(result.attributes.get("label", "")),
                "track_id": result.track_ids[0] if result.track_ids else "",
                "event_id": result.event_id,
            }
            for result in raw_results
        ]

    def list_metadata(self) -> List[Path]:
        return sorted(self.base_dir.glob("*.json"))

    def load_metadata(self, metadata_path: str | Path) -> EventVideoMetadata | None:
        path = Path(metadata_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return EventVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load event metadata: %s", path)
            return None

    def _expand_event_types(self, query: str) -> List[str]:
        if self.query_rewrite_service is not None:
            intent = self.query_rewrite_service.parse_query_intent(query)
            intent_event_types = [self._normalize_event_type(value) for value in intent.event_types]
            if intent_event_types:
                return intent_event_types

        normalized_query = self._normalize_event_type(query)
        results: List[str] = []
        for canonical, aliases in EVENT_TYPE_ALIASES.items():
            normalized_aliases = [self._normalize_event_type(alias) for alias in aliases]
            if normalized_query in normalized_aliases:
                results.append(self._normalize_event_type(canonical))
        if not results and normalized_query:
            results.append(normalized_query)
        return results

    def _normalize_event_type(self, value: str) -> str:
        return re.sub(r"\s+", "_", value.strip().casefold())

    def _extract_video_id(self, event_id: str) -> str:
        return event_id.split(":", 1)[0] if ":" in event_id else ""
