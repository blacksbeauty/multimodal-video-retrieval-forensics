from __future__ import annotations

from math import hypot
from typing import Any, Dict, List

from core.schemas import DetectionVideoMetadata, EventMetadata, TrajectoryVideoMetadata
from services.event_plugins.base import EventPluginBase, label_to_chinese
from services.event_plugins.geometry import (
    normalize_vector,
    point_in_polygon,
    trajectory_displacement,
    trajectory_main_direction,
)
from services.event_plugins.registry import register


class WrongWayDriving(EventPluginBase):
    """Emit events when a vehicle moves opposite to the configured lane direction."""

    plugin_name = "wrong_way_driving"
    event_type = "wrong_way_driving"

    def execute(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        config: Dict[str, Any],
    ) -> List[EventMetadata]:
        if not bool(config.get("enabled", True)):
            return []

        allowed_labels = set(config.get("allowed_labels", ["car", "truck", "bus", "motorcycle"]))
        allowed_direction = normalize_vector(config.get("allowed_direction", [1.0, 0.0]))
        if allowed_direction is None:
            return []
        min_track_points = max(int(config.get("min_track_points", 3)), 2)
        min_duration_sec = max(float(config.get("min_duration_sec", 1.0)), 0.0)
        min_displacement_px = max(float(config.get("min_displacement_px", 20.0)), 0.0)
        max_direction_dot = float(config.get("max_direction_dot", -0.3))
        roi_polygon = config.get("roi_polygon", [])
        min_roi_ratio = min(max(float(config.get("min_roi_ratio", 0.5)), 0.0), 1.0)

        events: List[EventMetadata] = []
        for track in trajectories.tracks:
            if track.label not in allowed_labels or len(track.points) < min_track_points:
                continue
            if track.duration_sec < min_duration_sec or trajectory_displacement(track.points) < min_displacement_px:
                continue

            first, last = track.points[0], track.points[-1]
            # Robust main direction: middle-segment displacement resists
            # head/tail jitter and mid-track U-turns (M2 fix).
            motion = trajectory_main_direction(
                track.points,
                start_frac=float(config.get("direction_start_frac", 0.25)),
                end_frac=float(config.get("direction_end_frac", 0.75)),
            )
            if motion is None:
                motion = normalize_vector((last.center_x - first.center_x, last.center_y - first.center_y))
            if motion is None:
                continue
            direction_dot = motion[0] * allowed_direction[0] + motion[1] * allowed_direction[1]
            if direction_dot > max_direction_dot:
                continue

            roi_ratio = 1.0
            if roi_polygon:
                if not isinstance(roi_polygon, list) or len(roi_polygon) < 3:
                    continue
                inside_count = sum(
                    point_in_polygon((point.center_x, point.center_y), roi_polygon)
                    for point in track.points
                )
                roi_ratio = inside_count / len(track.points)
                if roi_ratio < min_roi_ratio:
                    continue

            evidence_frames = self._evidence_frames(track.points)
            events.append(
                EventMetadata(
                    # S7: 新格式 event_id = {video_id}:{event_type}:{n}（n 取 track 序号）
                    event_id=f"{video_id}:{self.plugin_name}:{track.track_id.rsplit(':', 1)[-1]}",
                    event_type=self.event_type,
                    plugin_name=self.plugin_name,
                    video_id=video_id,
                    video_name=trajectories.video_name,
                    video_path=trajectories.video_path,
                    start_ts=track.start_ts,
                    end_ts=track.end_ts,
                    track_ids=[track.track_id],
                    confidence=min(1.0, float(track.avg_confidence) * (1.0 + min(abs(direction_dot), 1.0)) / 2.0),
                    representative_frame=track.representative_frame,
                    evidence_frames=evidence_frames,
                    attributes={
                        "label": track.label,
                        "direction": track.direction,
                        "allowed_direction": list(allowed_direction),
                        "direction_dot": direction_dot,
                        "roi_ratio": roi_ratio,
                        "min_displacement_px": min_displacement_px,
                    },
                    # S7: 中文 description
                    description=f"{label_to_chinese(track.label)}沿车道反向行驶（逆行）",
                )
            )
        return events

    def _evidence_frames(self, points) -> List[str]:
        if not points:
            return []
        candidates = (points[0], points[len(points) // 2], points[-1])
        evidence: List[str] = []
        for point in candidates:
            if point.frame_path not in evidence:
                evidence.append(point.frame_path)
        return evidence


register(WrongWayDriving.plugin_name, WrongWayDriving)
