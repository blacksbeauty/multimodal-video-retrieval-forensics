from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from config import Settings


logger = logging.getLogger(__name__)


class SegmentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def aggregate_frames(
        self,
        frames: Sequence[Dict],
        time_window_seconds: float = 1.5,
    ) -> List[Dict]:
        if time_window_seconds < 0:
            raise ValueError("time_window_seconds must be greater than or equal to 0.")
        if not frames:
            return []

        normalized_frames = sorted(
            [self._normalize_frame(frame) for frame in frames],
            key=lambda item: (item["video_name"], item["timestamp"], item["frame_path"]),
        )

        segments: List[Dict] = []
        current_segment: Dict | None = None

        for frame in normalized_frames:
            if current_segment is None:
                current_segment = self._new_segment(frame)
                continue

            if self._can_merge(current_segment, frame, time_window_seconds):
                current_segment["end_time"] = frame["timestamp"]
                current_segment["frames"].append(frame)
            else:
                segments.append(current_segment)
                current_segment = self._new_segment(frame)

        if current_segment is not None:
            segments.append(current_segment)

        logger.info(
            "Aggregated %s frames into %s segments with window=%s",
            len(normalized_frames),
            len(segments),
            time_window_seconds,
        )
        return segments

    def _normalize_frame(self, frame: Dict) -> Dict:
        if "video_name" not in frame:
            raise ValueError("Each frame must contain video_name.")
        if "frame_path" not in frame:
            raise ValueError("Each frame must contain frame_path.")

        if "timestamp" in frame:
            timestamp = float(frame["timestamp"])
        elif "timestamp_seconds" in frame:
            timestamp = float(frame["timestamp_seconds"])
        else:
            raise ValueError("Each frame must contain timestamp or timestamp_seconds.")

        return {
            "video_name": str(frame["video_name"]),
            "timestamp": timestamp,
            "score": float(frame.get("score", 0.0)),
            "frame_path": str(frame["frame_path"]),
        }

    def _new_segment(self, frame: Dict) -> Dict:
        return {
            "video_name": frame["video_name"],
            "start_time": frame["timestamp"],
            "end_time": frame["timestamp"],
            "frames": [frame],
        }

    def _can_merge(self, current_segment: Dict, frame: Dict, time_window_seconds: float) -> bool:
        if current_segment["video_name"] != frame["video_name"]:
            return False

        gap = frame["timestamp"] - current_segment["end_time"]
        return gap <= time_window_seconds
