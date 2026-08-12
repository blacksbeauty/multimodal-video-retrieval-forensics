from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.schemas import DetectionVideoMetadata, EventMetadata, TrajectoryPoint, TrajectoryVideoMetadata
from services.event_plugins.base import EventPluginBase
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
                    event_id=f"{video_id}:{self.plugin_name}:{track.track_id}",
                    event_type=self.event_type,
                    plugin_name=self.plugin_name,
                    video_id=video_id,
                    video_name=trajectories.video_name,
                    video_path=trajectories.video_path,
                    start_ts=track.start_ts,
                    end_ts=track.end_ts,
                    track_ids=[track.track_id],
                    confidence=min(track.avg_confidence, 1.0),
                    representative_frame=track.representative_frame,
                    evidence_frames=evidence_frames,
                    attributes={
                        "label": track.label,
                        "direction": track.direction,
                        "line": line,
                        "cross_timestamp": crossing["timestamp"],
                        "crossing_mode": crossing["mode"],
                    },
                    description=f"{track.label} crosses configured line",
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
