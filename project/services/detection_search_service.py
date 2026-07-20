from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from config import Settings
from core.schemas import DetectionSearchResult, DetectionVideoMetadata


logger = logging.getLogger(__name__)
DETECTION_LABELS = {"car", "truck", "bus", "motorcycle", "person", "traffic light"}


class DetectionSearchService:
    """Search persisted traffic detection metadata by exact label match."""

    def __init__(self, settings: Settings, query_rewrite_service=None) -> None:
        self.settings = settings
        self.base_dir = self.settings.detection_metadata_dir
        self.query_rewrite_service = query_rewrite_service
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def search_objects(self, query: str, top_k: int = 10) -> List[DetectionSearchResult]:
        """Search detection metadata by normalized object label."""
        intent = self._parse_intent(query)
        normalized_queries = [self._normalize_entity_name(label) for label in intent.primary_entities if label]
        if intent.query_type == "event":
            return []
        if not normalized_queries:
            raise ValueError("Detection search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        results: List[DetectionSearchResult] = []
        metadata_files = self.list_metadata()
        if not metadata_files:
            logger.warning("No detection metadata files available for detection search.")
            return []

        for metadata_path in metadata_files:
            metadata = self.load_metadata(metadata_path)
            if metadata is None:
                continue

            for frame in metadata.frames:
                detections = list(frame.detections)
                primary_detections = [
                    detection
                    for detection in detections
                    if self.normalize_label(detection.label) in normalized_queries
                ]
                if not primary_detections:
                    continue
                if not self._frame_matches_intent(primary_detections, detections, intent):
                    continue

                for detection in primary_detections:
                    if self.normalize_label(detection.label) not in normalized_queries:
                        continue
                    results.append(
                        DetectionSearchResult(
                            video_name=frame.video_name,
                            timestamp=frame.timestamp,
                            frame_path=frame.frame_path,
                            matched_label=detection.label,
                            confidence=detection.confidence,
                        )
                    )

        ordered = sorted(results, key=lambda item: (-item.confidence, item.video_name, item.timestamp))
        logger.info(
            "Completed detection search queries=%s results=%s",
            normalized_queries,
            len(ordered),
        )
        return ordered[:top_k]

    def search_as_frames(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search detection metadata and return frame-like records for hybrid fusion."""
        raw_results = self.search_objects(query=query, top_k=top_k)
        return [
            {
                "video_id": self._find_video_id_for_frame(result.frame_path),
                "video_name": result.video_name,
                "video_path": self._find_video_path_for_frame(result.frame_path),
                "frame_id": f"detection::{result.frame_path}",
                "frame_path": result.frame_path,
                "thumbnail_frame": result.frame_path,
                "timestamp_seconds": result.timestamp,
                "score": float(result.confidence),
                "detection_score": float(result.confidence),
                "matched_by": ["detection"],
                "matched_label": result.matched_label,
            }
            for result in raw_results
        ]

    def normalize_label(self, label: str) -> str:
        """Normalize a detection label query to lowercase and collapsed spaces."""
        return re.sub(r"\s+", " ", label.strip().casefold())

    def list_metadata(self) -> List[Path]:
        """List all persisted detection metadata files."""
        return sorted(self.base_dir.glob("*.json"))

    def load_metadata(self, metadata_path: str | Path) -> DetectionVideoMetadata | None:
        """Load one detection metadata file and validate its payload."""
        path = Path(metadata_path)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return DetectionVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load detection metadata: %s", path)
            return None

    def _find_video_id_for_frame(self, frame_path: str) -> str:
        metadata = self._lookup_metadata_by_frame(frame_path)
        return metadata.video_id if metadata is not None else ""

    def _find_video_path_for_frame(self, frame_path: str) -> str:
        metadata = self._lookup_metadata_by_frame(frame_path)
        return metadata.video_path if metadata is not None else ""

    def _lookup_metadata_by_frame(self, frame_path: str) -> DetectionVideoMetadata | None:
        for metadata_path in self.list_metadata():
            metadata = self.load_metadata(metadata_path)
            if metadata is None:
                continue
            if any(frame.frame_path == frame_path for frame in metadata.frames):
                return metadata
        return None

    def _expand_queries(self, query: str) -> List[str]:
        """Expand a raw query into normalized detection labels."""
        return [self.normalize_label(label) for label in self._parse_intent(query).primary_entities if label]

    def _parse_intent(self, query: str):
        if self.query_rewrite_service is None:
            value = self.normalize_label(query)
            from core.schemas import QueryIntent

            return QueryIntent(primary_entities=[value] if value else [], rewritten_queries=[query] if query else [])
        return self.query_rewrite_service.parse_query_intent(query)

    def _frame_matches_intent(self, primary_detections, all_detections, intent) -> bool:
        context_labels = [self._normalize_entity_name(entity) for entity in intent.context_entities]
        detectable_context = [label for label in context_labels if label in DETECTION_LABELS]
        if not detectable_context:
            return True

        context_detections = [
            detection
            for detection in all_detections
            if self.normalize_label(detection.label) in detectable_context
        ]
        if not context_detections:
            return False

        relation_set = set(intent.relations)
        if "near" in relation_set:
            return any(self._boxes_near(primary.bbox, context.bbox) for primary in primary_detections for context in context_detections)
        if "left_of" in relation_set:
            return any(self._is_left_of(primary.bbox, context.bbox) for primary in primary_detections for context in context_detections)
        if "right_of" in relation_set:
            return any(self._is_right_of(primary.bbox, context.bbox) for primary in primary_detections for context in context_detections)
        if "inside" in relation_set:
            return any(self._is_inside(primary.bbox, context.bbox) for primary in primary_detections for context in context_detections)

        return True

    def _normalize_entity_name(self, entity: str) -> str:
        return self.normalize_label(entity.replace("_", " "))

    def _boxes_near(self, primary_bbox, context_bbox) -> bool:
        if len(primary_bbox) != 4 or len(context_bbox) != 4:
            return True
        primary_center = ((primary_bbox[0] + primary_bbox[2]) / 2.0, (primary_bbox[1] + primary_bbox[3]) / 2.0)
        context_center = ((context_bbox[0] + context_bbox[2]) / 2.0, (context_bbox[1] + context_bbox[3]) / 2.0)
        dx = primary_center[0] - context_center[0]
        dy = primary_center[1] - context_center[1]
        distance_sq = dx * dx + dy * dy
        span = max(primary_bbox[2] - primary_bbox[0], context_bbox[2] - context_bbox[0], 1.0)
        return distance_sq <= (span * 2.5) ** 2

    def _is_left_of(self, primary_bbox, context_bbox) -> bool:
        if len(primary_bbox) != 4 or len(context_bbox) != 4:
            return False
        return ((primary_bbox[0] + primary_bbox[2]) / 2.0) < ((context_bbox[0] + context_bbox[2]) / 2.0)

    def _is_right_of(self, primary_bbox, context_bbox) -> bool:
        if len(primary_bbox) != 4 or len(context_bbox) != 4:
            return False
        return ((primary_bbox[0] + primary_bbox[2]) / 2.0) > ((context_bbox[0] + context_bbox[2]) / 2.0)

    def _is_inside(self, primary_bbox, context_bbox) -> bool:
        if len(primary_bbox) != 4 or len(context_bbox) != 4:
            return False
        return (
            primary_bbox[0] >= context_bbox[0]
            and primary_bbox[1] >= context_bbox[1]
            and primary_bbox[2] <= context_bbox[2]
            and primary_bbox[3] <= context_bbox[3]
        )
