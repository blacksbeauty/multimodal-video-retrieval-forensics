from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2

from core.schemas import DetectionFrameMetadata, DetectionVideoMetadata, EventMetadata, TrajectoryPoint, TrajectoryVideoMetadata
from services.event_plugins.base import EventPluginBase, label_to_chinese
from services.event_plugins.geometry import find_line_contact, trajectory_displacement
from services.event_plugins.registry import register


class RedLightViolation(EventPluginBase):
    """Emit red-light-violation events when vehicles cross the stop line during a red phase."""

    plugin_name = "red_light_violation"
    event_type = "red_light_violation"

    def execute(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        config: Dict[str, Any],
    ) -> List[EventMetadata]:
        if not bool(config.get("enabled", True)):
            return []

        stop_line = config.get("line")
        if not self._valid_line(stop_line):
            return []

        allowed_labels = set(config.get("allowed_labels", ["car", "truck", "bus", "motorcycle"]))
        state_window_sec = float(config.get("state_window_sec", 2.0))
        min_red_duration_sec = float(config.get("min_red_duration_sec", 0.0))
        min_after_crossing_points = max(int(config.get("min_after_crossing_points", 2)), 1)
        min_after_crossing_displacement_px = float(config.get("min_after_crossing_displacement_px", 15.0))
        traffic_light_states = self._extract_traffic_light_states(
            detections.frames,
            min_confidence=float(config.get("traffic_light_min_confidence", 0.25)),
            min_state_score=float(config.get("traffic_light_min_state_score", 0.01)),
        )
        if not traffic_light_states:
            return []

        events: List[EventMetadata] = []
        for track in trajectories.tracks:
            if track.label not in allowed_labels:
                continue

            crossing = find_line_contact(
                track.points,
                stop_line,
                min_displacement_px=float(config.get("min_displacement_px", 10.0)),
            )
            if crossing is None:
                continue

            # H2 fix: verify the vehicle actually keeps moving after crossing
            # the stop line, so red-light *stopping on the line* (legal) is
            # not mistaken for running a red light.
            crossing_index = int(crossing["index"])
            after_points = track.points[crossing_index + 1 :]
            if len(after_points) < min_after_crossing_points:
                continue
            crossing_point = track.points[crossing_index]
            if trajectory_displacement([crossing_point] + after_points) < min_after_crossing_displacement_px:
                continue

            light_state = self._state_at_timestamp(
                traffic_light_states,
                crossing["timestamp"],
                max_delta_sec=state_window_sec,
            )
            if light_state is None or light_state["state"] != "red":
                continue
            # M3 fix: the light must have been steadily red around the
            # crossing moment; a single red-frame blip is not enough.
            if not self._in_sustained_red(
                traffic_light_states,
                crossing["timestamp"],
                min_red_duration_sec,
                merge_gap_sec=float(config.get("red_merge_gap_sec", 2.0)),
            ):
                continue

            evidence_frames = self._evidence_frames(track.points, crossing["index"])
            if light_state["frame_path"] not in evidence_frames:
                evidence_frames.insert(0, light_state["frame_path"])

            confidence = min(
                1.0,
                float(track.avg_confidence) * 0.6 + float(light_state["confidence"]) * 0.4,
            )
            events.append(
                EventMetadata(
                    # S7: 新格式 event_id = {video_id}:{event_type}:{n}（n 取 track 序号，
                    # 与迁移脚本产出格式一致，数据规范 v1.1 §2）
                    event_id=f"{video_id}:{self.plugin_name}:{track.track_id.rsplit(':', 1)[-1]}",
                    event_type=self.event_type,
                    plugin_name=self.plugin_name,
                    video_id=video_id,
                    video_name=trajectories.video_name,
                    video_path=trajectories.video_path,
                    start_ts=max(crossing["timestamp"] - 1.0, track.start_ts),
                    end_ts=min(crossing["timestamp"] + 1.0, track.end_ts),
                    track_ids=[track.track_id],
                    confidence=confidence,
                    representative_frame=track.representative_frame,
                    evidence_frames=evidence_frames,
                    attributes={
                        "label": track.label,
                        "direction": track.direction,
                        "line": stop_line,
                        "cross_timestamp": crossing["timestamp"],
                        "crossing_mode": crossing["mode"],
                        "light_state": light_state["state"],
                        "light_state_confidence": light_state["confidence"],
                        "light_frame": light_state["frame_path"],
                    },
                    # S7: 中文 description（供 segment.text 模板直接用，提升中文检索语义）
                    description=f"{label_to_chinese(track.label)}在红灯状态下越过停止线",
                )
            )

        return events

    def _extract_traffic_light_states(
        self,
        frames: Sequence[DetectionFrameMetadata],
        min_confidence: float,
        min_state_score: float,
    ) -> List[Dict[str, Any]]:
        states: List[Dict[str, Any]] = []
        for frame in frames:
            best_detection = None
            for detection in frame.detections:
                if detection.label != "traffic light":
                    continue
                if detection.confidence < min_confidence or len(detection.bbox) != 4:
                    continue
                if best_detection is None or detection.confidence > best_detection.confidence:
                    best_detection = detection

            if best_detection is None:
                continue

            state_payload = self._classify_traffic_light_state(
                frame_path=frame.frame_path,
                bbox=best_detection.bbox,
                min_state_score=min_state_score,
            )
            if state_payload is None:
                continue

            states.append(
                {
                    "timestamp": frame.timestamp,
                    "frame_path": frame.frame_path,
                    "state": state_payload["state"],
                    "confidence": state_payload["confidence"],
                }
            )

        return states

    def _classify_traffic_light_state(
        self,
        frame_path: str,
        bbox: Sequence[float],
        min_state_score: float,
    ) -> Dict[str, float | str] | None:
        image = cv2.imread(frame_path)
        if image is None:
            return None

        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        height = hsv.shape[0]
        top = hsv[: max(height // 3, 1)]
        middle = hsv[height // 3 : max((2 * height) // 3, height // 3 + 1)]
        bottom = hsv[max((2 * height) // 3, 0) :]

        red_score = self._color_score(hsv, top, ((0, 80, 80), (10, 255, 255)), ((160, 80, 80), (180, 255, 255)))
        yellow_score = self._color_score(hsv, middle, ((15, 70, 80), (40, 255, 255)))
        green_score = self._color_score(hsv, bottom, ((40, 60, 60), (95, 255, 255)))

        scores = {
            "red": red_score,
            "yellow": yellow_score,
            "green": green_score,
        }
        state, score = max(scores.items(), key=lambda item: item[1])
        if score < min_state_score:
            return None

        return {
            "state": state,
            "confidence": min(score * 8.0, 1.0),
        }

    def _color_score(
        self,
        full_hsv,
        weighted_hsv,
        *ranges: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> float:
        if full_hsv.size == 0 or weighted_hsv.size == 0:
            return 0.0

        overall_mask = None
        weighted_mask = None
        for lower, upper in ranges:
            current_overall = cv2.inRange(full_hsv, lower, upper)
            current_weighted = cv2.inRange(weighted_hsv, lower, upper)
            overall_mask = current_overall if overall_mask is None else cv2.bitwise_or(overall_mask, current_overall)
            weighted_mask = current_weighted if weighted_mask is None else cv2.bitwise_or(weighted_mask, current_weighted)

        overall_ratio = float(cv2.countNonZero(overall_mask)) / float(overall_mask.size)
        weighted_ratio = float(cv2.countNonZero(weighted_mask)) / float(weighted_mask.size)
        return overall_ratio * 0.4 + weighted_ratio * 0.6

    def _state_at_timestamp(
        self,
        states: Sequence[Dict[str, Any]],
        timestamp: float,
        max_delta_sec: float,
    ) -> Dict[str, Any] | None:
        candidates = [
            state
            for state in states
            if abs(float(state["timestamp"]) - timestamp) <= max_delta_sec
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda item: (abs(float(item["timestamp"]) - timestamp), -float(item["confidence"])))
        return candidates[0]

    def _red_phase_intervals(
        self,
        states: Sequence[Dict[str, Any]],
        merge_gap_sec: float = 2.0,
    ) -> List[tuple[float, float]]:
        """Merge consecutive red-light states into continuous red intervals.

        ``merge_gap_sec`` must tolerate the frame-sampling interval (e.g. 1s
        when sampling at 1 fps), otherwise a steadily red light would be split
        into isolated single-frame intervals.
        """
        red_times = sorted(float(state["timestamp"]) for state in states if state["state"] == "red")
        if not red_times:
            return []
        intervals: List[tuple[float, float]] = []
        interval_start = previous = red_times[0]
        for current in red_times[1:]:
            if current - previous > merge_gap_sec:
                intervals.append((interval_start, previous))
                interval_start = current
            previous = current
        intervals.append((interval_start, previous))
        return intervals

    def _in_sustained_red(
        self,
        states: Sequence[Dict[str, Any]],
        timestamp: float,
        min_red_duration_sec: float,
        merge_gap_sec: float = 2.0,
    ) -> bool:
        """Return whether *timestamp* falls inside a continuous red interval
        whose duration is at least ``min_red_duration_sec``.

        A value <= 0 disables the sustained-red check (legacy single-point
        matching behaviour).
        """
        if min_red_duration_sec <= 0:
            return True
        for start, end in self._red_phase_intervals(states, merge_gap_sec):
            if start <= timestamp <= end + 1e-6 and (end - start) >= min_red_duration_sec:
                return True
        return False

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


register(RedLightViolation.plugin_name, RedLightViolation)
