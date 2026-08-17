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
    "vehicle_crosses_line": ("vehicle_crosses_line", "cross line", "line crossing", "压线", "车辆压线", "压线行驶", "跨线", "跨线行驶"),
    "wrong_way_driving": ("wrong_way_driving", "wrong way driving", "wrong way", "逆行", "车辆逆行", "反向行驶", "逆向行驶", "逆方向行驶", "反方向", "opposite direction"),
    "red_light_violation": ("red_light_violation", "red light violation", "red light", "闯红灯", "车辆闯红灯", "冲红灯", "红灯违规", "红灯违章", "红灯违法", "违反红灯", "running red light"),
}


class EventSearchService:
    """Search generated event metadata using deterministic event type matching."""

    def __init__(self, settings: Settings, query_rewrite_service=None) -> None:
        self.settings = settings
        self.base_dir = self.settings.event_metadata_dir
        self.query_rewrite_service = query_rewrite_service
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 元数据 mtime 缓存（Code Review Nice to Have）：避免每次检索全量重读 JSON。
        self._cache: Dict[str, tuple[float, EventVideoMetadata]] = {}

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
                        key_snapshots=event.key_snapshots,
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
                "key_snapshots": list(result.key_snapshots),
            }
            for result in raw_results
        ]

    def list_metadata(self) -> List[Path]:
        return sorted(self.base_dir.glob("*.json"))

    def load_metadata(self, metadata_path: str | Path) -> EventVideoMetadata | None:
        """Load one event metadata file and validate its payload.

        带 mtime 校验缓存：文件未变化时直接复用内存对象，避免检索路径重复全量读盘。
        """
        path = Path(metadata_path)
        if not path.exists():
            return None

        cache_key = str(path)
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = EventVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load event metadata: %s", path)
            return None

        if len(self._cache) > 1024:
            self._cache.clear()
        self._cache[cache_key] = (current_mtime, metadata)
        return metadata

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
