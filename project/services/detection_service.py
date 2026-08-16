from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Dict, List

import cv2

from config import Settings
from core.schemas import DetectionFrameMetadata, DetectionItem, DetectionVideoMetadata
from utils.path_utils import build_asset_id, normalize_path


logger = logging.getLogger(__name__)

SUPPORTED_TRAFFIC_LABELS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "person",
    "traffic light",
}


class DetectionService:
    """Run YOLOv8 object detection on extracted traffic video frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.model_load_attempted = False
        self.model_available = False
        self.model_load_error: str | None = None
        self.settings.detection_metadata_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> bool:
        """Load YOLOv8 once in CPU mode and cache the result."""
        if self.model_available and self.model is not None:
            return True

        if self.model_load_attempted and not self.model_available:
            return False

        self.model_load_attempted = True

        try:
            from ultralytics import YOLO
        except ImportError:
            self.model_load_error = "ultralytics is not installed. Please install ultralytics to enable detection."
            logger.exception(self.model_load_error)
            return False

        try:
            logger.info("Loading YOLOv8 model=%s on cpu", self.settings.detection_model_name)
            self.model = YOLO(self.settings.detection_model_name)
            self.model_available = True
            self.model_load_error = None
            return True
        except Exception as exc:
            self.model_load_error = f"Failed to initialize YOLOv8 model: {exc}"
            logger.exception("Failed to load YOLOv8 model.")
            self.model = None
            self.model_available = False
            return False

    def detect_frame(self, frame_path: str | Path) -> DetectionFrameMetadata | None:
        """Run detection on a single frame and return structured metadata."""
        if not self.load_model():
            return None

        resolved_frame_path = Path(frame_path).expanduser().resolve()
        metadata_frame_path = self._resolve_metadata_frame_path(resolved_frame_path)
        image = cv2.imread(str(resolved_frame_path))
        if image is None:
            logger.warning("Skipping unreadable frame during detection: %s", resolved_frame_path)
            return None

        try:
            results = self.model.predict(
                source=image,
                conf=self.settings.detection_score_threshold,
                device="cpu",
                verbose=False,
            )
        except Exception:
            logger.exception("Detection failed for frame %s", resolved_frame_path)
            return None

        detections: List[DetectionItem] = []
        for result in results:
            detections.extend(self._extract_detection_items(result))

        if not detections:
            logger.debug("No supported traffic detections found for frame %s", resolved_frame_path.name)

        return DetectionFrameMetadata(
            video_name=self._extract_video_name_from_frame(metadata_frame_path),
            frame_path=normalize_path(metadata_frame_path),
            timestamp=self._extract_timestamp_from_frame_name(resolved_frame_path),
            detections=detections,
        )

    def process_frames_directory(self, directory: str | Path | None = None) -> List[DetectionVideoMetadata]:
        """Process every frame in a directory and save grouped detection metadata per video."""
        target_directory = Path(directory).expanduser().resolve() if directory else self.settings.frames_dir.resolve()
        if not target_directory.exists() or not target_directory.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {target_directory}")

        frame_paths = sorted(target_directory.glob("*.jpg"))
        if not frame_paths:
            logger.warning("No frame images found in %s", target_directory)
            return []

        grouped_frames: Dict[str, List[DetectionFrameMetadata]] = {}
        logger.info("Starting detection batch processing for %s frames", len(frame_paths))

        for frame_path in frame_paths:
            frame_metadata = self.detect_frame(frame_path)
            if frame_metadata is None:
                continue
            video_key = self._extract_video_name_from_frame(frame_path)
            grouped_frames.setdefault(video_key, []).append(frame_metadata)

        videos: List[DetectionVideoMetadata] = []
        for video_name, frames in grouped_frames.items():
            if not frames:
                continue

            try:
                video_path = self._resolve_video_path_from_frame_group(video_name)
            except FileNotFoundError as exc:
                # Code Review Must Fix #4：源视频缺失时跳过该视频并记录日志，
                # 不写假路径、不中断整批处理。
                logger.warning("Skipping detection metadata for missing video: %s", exc)
                continue
            video_id = build_asset_id(video_path)
            canonical_video_name = video_path.name
            ordered_frames = sorted(frames, key=lambda item: item.timestamp)
            for frame in ordered_frames:
                frame.video_name = canonical_video_name
            video_metadata = DetectionVideoMetadata(
                video_id=video_id,
                video_name=canonical_video_name,
                video_path=normalize_path(video_path),
                frames=ordered_frames,
            )
            self.save_detection_metadata(video_metadata)
            videos.append(video_metadata)

        logger.info(
            "Completed detection batch processing videos=%s processed_frames=%s",
            len(videos),
            sum(len(video.frames) for video in videos),
        )
        return videos

    def save_detection_metadata(self, detection_metadata: DetectionVideoMetadata) -> Path:
        """Persist detection metadata under metadata/detections/<video_id>.json."""
        output_path = self.settings.detection_metadata_dir / f"{detection_metadata.video_id}.json"
        payload = detection_metadata.model_dump_json(indent=2)
        temp_output_path = output_path.with_name(f"{output_path.name}.{time.time_ns()}.tmp")
        temp_output_path.write_text(payload, encoding="utf-8")

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                temp_output_path.replace(output_path)
                break
            except PermissionError as exc:
                last_error = exc
                logger.warning(
                    "Retrying detection metadata replace path=%s attempt=%s",
                    output_path,
                    attempt + 1,
                )
                time.sleep(0.2)
        else:
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)
            raise last_error or PermissionError(f"Failed to replace detection metadata file: {output_path}")

        logger.info("Saved detection metadata to %s", output_path)
        return output_path

    def _extract_detection_items(self, result) -> List[DetectionItem]:
        """Convert a YOLO result into traffic-filtered detection metadata items."""
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        cls_values = boxes.cls.tolist() if boxes.cls is not None else []
        conf_values = boxes.conf.tolist() if boxes.conf is not None else []
        bbox_values = boxes.xyxy.tolist() if boxes.xyxy is not None else []

        detections: List[DetectionItem] = []
        for cls_idx, confidence, bbox in zip(cls_values, conf_values, bbox_values):
            normalized_class_id = int(cls_idx)
            label = str(names.get(normalized_class_id, "")).strip().casefold()
            if label not in SUPPORTED_TRAFFIC_LABELS:
                continue

            detections.append(
                DetectionItem(
                    label=label,      # S5: 双写别名（过渡期，v2.0 移除）
                    class_=label,     # S5: 规范标准字段，值相同
                    confidence=float(confidence),
                    bbox=[float(value) for value in bbox],
                    class_id=normalized_class_id,
                )
            )

        return detections

    def _resolve_metadata_frame_path(self, frame_path: Path) -> Path:
        """Resolve the canonical frame path to persist in detection metadata.

        Detection may be executed over copied frames in a temporary directory.
        When a same-named frame exists in the project's managed frames directory,
        persist that stable path instead of the transient processing path.
        """
        canonical_candidate = (self.settings.frames_dir / frame_path.name).resolve()
        if canonical_candidate.exists() and canonical_candidate.is_file():
            if canonical_candidate != frame_path:
                logger.debug(
                    "Canonicalized detection frame path from %s to %s",
                    frame_path,
                    canonical_candidate,
                )
            return canonical_candidate

        return frame_path.resolve()

    def _extract_timestamp_from_frame_name(self, frame_path: Path) -> float:
        stem_parts = frame_path.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            return 0.0

        try:
            return float(stem_parts[1])
        except ValueError:
            return 0.0

    def _extract_video_name_from_frame(self, frame_path: Path) -> str:
        stem_parts = frame_path.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            return f"{frame_path.stem}.mp4"
        return f"{stem_parts[0]}.mp4"

    def _resolve_video_path_from_frame_group(self, video_name: str) -> Path:
        imported_video_path = self._lookup_imported_video_path(video_name)
        if imported_video_path is not None:
            return imported_video_path

        stem = Path(video_name).stem
        candidates = [
            self.settings.videos_dir / video_name,
            self.settings.videos_dir / f"{stem}.avi",
            self.settings.videos_dir / f"{stem}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        # Code Review Must Fix #4：找不到源视频时抛异常，而不是返回一个
        # 不存在的假路径写入元数据（会导致 clip_available 误判、片段下载 404）。
        raise FileNotFoundError(
            f"Source video not found for {video_name!r}; searched: "
            f"{[str(c) for c in candidates]}"
        )

    def _lookup_imported_video_path(self, video_name: str) -> Path | None:
        """Resolve a logical imported sequence name back to its original dataset directory."""
        source_map_path = self.settings.dataset_metadata_dir / "streetscene_sources.json"
        if not source_map_path.exists():
            return None

        try:
            payload = json.loads(source_map_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load StreetScene source map for detection path lookup.")
            return None

        entry = payload.get(video_name)
        if not isinstance(entry, dict):
            return None

        source_sequence_dir = entry.get("source_sequence_dir")
        if not source_sequence_dir:
            return None

        return Path(source_sequence_dir).expanduser().resolve()
