from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

from config import Settings


logger = logging.getLogger(__name__)


class SegmentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_segment_text(
        self,
        video_id: str,
        start_ts: float,
        description: str = "",
        class_summary: str = "",
        ocr_text: str = "",
    ) -> str:
        """S2: 段检索文本生成（数据规范 v1.1 §3.8 text 模板）。

        优先级:
          1. 事件段: 直接用事件 description（如 "车辆逆行，置信度0.92"）
          2. 兜底段: "检测到 {classes}，识别到文本 {ocr_text}"
          3. 全空兜底: "视频片段_{video_id}_{start_ts}"
        禁止返回空字符串。
        """
        if description and description.strip():
            return description.strip()
        if class_summary.strip() or ocr_text.strip():
            parts = []
            if class_summary.strip():
                parts.append(f"检测到 {class_summary.strip()}")
            if ocr_text.strip():
                parts.append(f"识别到文本 {ocr_text.strip()}")
            return "，".join(parts)
        return f"视频片段_{video_id}_{start_ts}"

    def persist_segments(
        self,
        video_id: str,
        segments: Sequence[Dict],
        embeddings_meta: Sequence[Dict],
    ) -> Path:
        """S4: 将段与段向量 meta 聚合落盘到 metadata/segments/{video_id}.json。

        文件结构（数据规范 v1.1 §3.8 + §3.9）:
        {
          "video_id": ...,
          "segments": [ {segment 协议...}, ... ],
          "embeddings": [ {segment_id, model, dimension, path, text_source, created_at}, ... ]
        }

        返回写入的文件路径。
        """
        output_dir = self.settings.segment_metadata_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.json"
        payload = {
            "video_id": video_id,
            "segments": [dict(item) for item in segments],
            "embeddings": [dict(item) for item in embeddings_meta],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Persisted segments for video_id=%s segments=%s embeddings=%s path=%s",
            video_id,
            len(segments),
            len(embeddings_meta),
            output_path,
        )
        return output_path

    def aggregate_frames(
        self,
        frames: Sequence[Dict],
        time_window_seconds: float | None = None,  # S6: None → 读 config.segment_window_seconds (5.0)
    ) -> List[Dict]:
        # S6: 兜底窗口统一由 config 控制，避免调用方传参不一致（数据规范 v1.1 §3.8）
        window = self.settings.segment_window_seconds if time_window_seconds is None else time_window_seconds
        if window < 0:
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

            if self._can_merge(current_segment, frame, window):
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
            window,
        )
        return segments

    def _normalize_frame(self, frame: Dict) -> Dict:
        if "video_name" not in frame:
            raise ValueError("Each frame must contain video_name.")
        if "frame_path" not in frame:
            raise ValueError("Each frame must contain frame_path.")

        # S3: 统一两位小数，保证与检测/轨迹/OCR 层 timestamp 可跨层对齐（数据规范 v1.1 §0）。
        if "timestamp" in frame:
            timestamp = round(float(frame["timestamp"]), 2)
        elif "timestamp_seconds" in frame:
            timestamp = round(float(frame["timestamp_seconds"]), 2)
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
            # S2: text 由上层(segment 生成流程)调用 build_segment_text() 填充，
            # 保证任意 segment.text 非空（数据规范 v1.1 §3.8 禁止空 text）。
            "text": "",
        }

    def _can_merge(self, current_segment: Dict, frame: Dict, time_window_seconds: float) -> bool:
        if current_segment["video_name"] != frame["video_name"]:
            return False

        gap = frame["timestamp"] - current_segment["end_time"]
        return gap <= time_window_seconds
