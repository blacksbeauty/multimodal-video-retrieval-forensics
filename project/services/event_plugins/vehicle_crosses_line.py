from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.schemas import DetectionVideoMetadata, EventMetadata, TrajectoryPoint, TrajectoryVideoMetadata
from services.event_plugins.base import EventPluginBase, label_to_chinese
from services.event_plugins.geometry import find_line_contact
from services.event_plugins.registry import register


class VehicleCrossesLine(EventPluginBase):
    """Emit events when a tracked vehicle crosses a configured virtual line."""

    plugin_name = "vehicle_crosses_line"
    event_type = "vehicle_crosses_line"

    def execute(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        config: Dict[str, Any],
    ) -> List[EventMetadata]:
        line = config.get("line")
        if not self._valid_line(line):
            return []

        allowed_labels = set(config.get("allowed_labels", ["car", "truck", "bus", "motorcycle"]))
        events: List[EventMetadata] = []
        for track in trajectories.tracks:
            if track.label not in allowed_labels:
                continue

            crossing = find_line_contact(
                track.points,
                line,
                min_displacement_px=float(config.get("min_displacement_px", 10.0)),
            )
            if crossing is None:
                continue

            evidence_frames = self._evidence_frames(track.points, crossing["index"])
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
                    confidence=min(track.avg_confidence, 1.0),
                    representative_frame=self.pick_representative_frame(
                        evidence_frames, track.representative_frame
                    ),
                    evidence_frames=evidence_frames,
                    key_snapshots=self.extract_three_keyframes(
                        points=track.points,
                        evidence_frames=evidence_frames,
                        line=config.get("line"),
                        anchor_timestamp=crossing["timestamp"],
                    ),
                    attributes={
                        "label": track.label,
                        "direction": track.direction,
                        "line": line,
                        "cross_timestamp": crossing["timestamp"],
                        "crossing_mode": crossing["mode"],
                    },
                    # S7: 中文 description
                    description=f"{label_to_chinese(track.label)}越过车道线（压线）",
                )
            )
        return events

    def _evidence_frames(self, points: Sequence[TrajectoryPoint], crossing_index: float) -> List[str]:
        index = int(crossing_index)
        evidence: List[str] = []
        for item in points[max(index - 1, 0) : min(index + 2, len(points))]:
            if item.frame_path not in evidence:
                evidence.append(item.frame_path)
        return evidence

    def _valid_line(self, line: Any) -> bool:
        return (
            isinstance(line, list)
            and len(line) == 2
            and all(isinstance(point, list) and len(point) == 2 for point in line)
        )


register(VehicleCrossesLine.plugin_name, VehicleCrossesLine)
