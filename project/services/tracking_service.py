from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Dict, List

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from config import Settings
from core.schemas import DetectionVideoMetadata, TrajectoryPoint, TrajectoryTrackMetadata, TrajectoryVideoMetadata


logger = logging.getLogger(__name__)


class TrackingService:
    """Build trajectory metadata from per-frame detection metadata using ByteTrack."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.trajectory_metadata_dir.mkdir(parents=True, exist_ok=True)

    def build_tracker(self, frame_rate: int) -> BYTETracker:
        """Create a ByteTrack tracker configured from application settings."""
        args = SimpleNamespace(
            track_high_thresh=self.settings.tracking_high_thresh,
            track_low_thresh=self.settings.tracking_low_thresh,
            new_track_thresh=self.settings.tracking_new_track_thresh,
            track_buffer=self.settings.tracking_track_buffer,
            match_thresh=self.settings.tracking_match_thresh,
            fuse_score=self.settings.tracking_fuse_score,
        )
        return BYTETracker(args=args, frame_rate=frame_rate)

    def process_detection_metadata_directory(
        self,
        directory: str | Path | None = None,
    ) -> List[TrajectoryVideoMetadata]:
        """Generate trajectory metadata for every detection metadata file in a directory."""
        target_directory = (
            Path(directory).expanduser().resolve()
            if directory
            else self.settings.detection_metadata_dir.resolve()
        )
        if not target_directory.exists() or not target_directory.is_dir():
            raise FileNotFoundError(f"Detection metadata directory not found: {target_directory}")

        metadata_paths = sorted(target_directory.glob("*.json"))
        if not metadata_paths:
            logger.warning("No detection metadata files found in %s", target_directory)
            return []

        outputs: List[TrajectoryVideoMetadata] = []
        logger.info("Starting tracking batch processing for %s detection metadata files", len(metadata_paths))
        for metadata_path in metadata_paths:
            trajectory_metadata = self.process_video_metadata(metadata_path)
            if trajectory_metadata is None:
                continue
            self.save_trajectory_metadata(trajectory_metadata)
            outputs.append(trajectory_metadata)

        logger.info("Completed tracking batch processing videos=%s", len(outputs))
        return outputs

    def process_video_metadata(self, metadata_path: str | Path) -> TrajectoryVideoMetadata | None:
        """Generate trajectory metadata for one detection metadata file."""
        detection_metadata = self._load_detection_metadata(Path(metadata_path))
        if detection_metadata is None:
            return None

        ordered_frames = sorted(detection_metadata.frames, key=lambda item: item.timestamp)
        frame_rate = self._estimate_frame_rate(ordered_frames)
        tracker = self.build_tracker(frame_rate=frame_rate)
        track_buffers: Dict[str, Dict[str, object]] = {}

        for frame in ordered_frames:
            track_rows = self._update_tracker(tracker, frame)
            class_id_to_label = {
                int(detection.class_id): detection.label
                for detection in frame.detections
                if detection.class_id is not None
            }

            for row in track_rows:
                if len(row) < 7:
                    continue

                bbox = [float(value) for value in row[:4]]
                track_identifier = f"{detection_metadata.video_id}:{int(row[4])}"
                confidence = float(row[5])
                class_id = int(row[6])
                label = class_id_to_label.get(class_id, f"class_{class_id}")
                center_x, center_y = self._bbox_center(bbox)
                point = TrajectoryPoint(
                    timestamp=float(frame.timestamp),
                    frame_path=frame.frame_path,
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    confidence=confidence,
                )

                buffer = track_buffers.setdefault(
                    track_identifier,
                    {
                        "label": label,
                        "points": [],
                    },
                )
                points: List[TrajectoryPoint] = buffer["points"]  # type: ignore[assignment]
                if points and points[-1].timestamp == point.timestamp:
                    continue
                points.append(point)

        tracks = self._finalize_tracks(track_buffers)
        return TrajectoryVideoMetadata(
            video_id=detection_metadata.video_id,
            video_name=detection_metadata.video_name,
            video_path=detection_metadata.video_path,
            tracks=tracks,
        )

    def save_trajectory_metadata(self, metadata: TrajectoryVideoMetadata) -> Path:
        """Persist trajectory metadata under metadata/trajectories/<video_id>.json."""
        output_path = self.settings.trajectory_metadata_dir / f"{metadata.video_id}.json"
        payload = metadata.model_dump_json(indent=2)
        temp_output_path = output_path.with_name(f"{output_path.name}.{time.time_ns()}.tmp")
        temp_output_path.write_text(payload, encoding="utf-8")
        temp_output_path.replace(output_path)
        logger.info("Saved trajectory metadata to %s", output_path)
        return output_path

    def _load_detection_metadata(self, metadata_path: Path) -> DetectionVideoMetadata | None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return DetectionVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load detection metadata for tracking: %s", metadata_path)
            return None

    def _estimate_frame_rate(self, frames) -> int:
        if len(frames) < 2:
            return self.settings.tracking_frame_rate

        deltas = [
            max(float(current.timestamp) - float(previous.timestamp), 0.0)
            for previous, current in zip(frames, frames[1:])
        ]
        positive_deltas = [delta for delta in deltas if delta > 0]
        if not positive_deltas:
            return self.settings.tracking_frame_rate

        median_delta = sorted(positive_deltas)[len(positive_deltas) // 2]
        estimated = max(int(round(1.0 / median_delta)), 1)
        return estimated

    def _update_tracker(self, tracker: BYTETracker, frame) -> np.ndarray:
        boxes_array = self._frame_to_tracker_array(frame)
        if boxes_array.size == 0:
            empty_boxes = Boxes(np.empty((0, 6), dtype=np.float32), (1, 1))
            return tracker.update(empty_boxes)

        orig_shape = self._infer_orig_shape(boxes_array[:, :4])
        boxes = Boxes(boxes_array, orig_shape)
        return tracker.update(boxes)

    def _frame_to_tracker_array(self, frame) -> np.ndarray:
        rows = []
        for detection in frame.detections:
            if len(detection.bbox) != 4 or detection.class_id is None:
                continue
            rows.append(
                [
                    float(detection.bbox[0]),
                    float(detection.bbox[1]),
                    float(detection.bbox[2]),
                    float(detection.bbox[3]),
                    float(detection.confidence),
                    float(detection.class_id),
                ]
            )

        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _infer_orig_shape(self, xyxy: np.ndarray) -> tuple[int, int]:
        max_x = max(int(np.max(xyxy[:, [0, 2]])), 1)
        max_y = max(int(np.max(xyxy[:, [1, 3]])), 1)
        return max_y + 1, max_x + 1

    def _finalize_tracks(self, track_buffers: Dict[str, Dict[str, object]]) -> List[TrajectoryTrackMetadata]:
        tracks: List[TrajectoryTrackMetadata] = []
        for track_id, payload in track_buffers.items():
            points = list(payload["points"])  # type: ignore[arg-type]
            if not points:
                continue

            ordered_points = sorted(points, key=lambda item: item.timestamp)
            confidences = [point.confidence for point in ordered_points]
            representative_point = max(ordered_points, key=lambda item: item.confidence)
            start_ts = float(ordered_points[0].timestamp)
            end_ts = float(ordered_points[-1].timestamp)

            tracks.append(
                TrajectoryTrackMetadata(
                    track_id=track_id,
                    label=str(payload["label"]),
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_sec=max(end_ts - start_ts, 0.0),
                    frame_count=len(ordered_points),
                    avg_confidence=float(sum(confidences) / len(confidences)),
                    max_confidence=float(max(confidences)),
                    direction=self._infer_direction(ordered_points),
                    representative_frame=representative_point.frame_path,
                    points=ordered_points,
                )
            )

        return sorted(tracks, key=lambda item: (item.label, item.start_ts, item.track_id))

    def _bbox_center(self, bbox: List[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _infer_direction(self, points: List[TrajectoryPoint]) -> str:
        if len(points) < 2:
            return "unknown"

        first = points[0]
        last = points[-1]
        dx = float(last.center_x - first.center_x)
        dy = float(last.center_y - first.center_y)
        threshold = self.settings.trajectory_direction_min_displacement

        if abs(dx) < threshold and abs(dy) < threshold:
            return "stationary"
        if abs(dx) >= abs(dy):
            return "left_to_right" if dx > 0 else "right_to_left"
        return "top_to_bottom" if dy > 0 else "bottom_to_top"
