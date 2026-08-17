from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def read_image_cv(frame_path: str | Path):
    """OpenCV 安全读图（Windows 中文路径兼容，Code Review 修复）。

    ``cv2.imread`` 在 Windows 上无法打开含非 ASCII（中文）字符的文件路径
    （内部用窄字符 fopen），返回 None。改用 ``np.fromfile + cv2.imdecode``
    从字节流解码，彻底规避路径编码问题。读取失败返回 None（与 imread 语义一致）。
    """
    import cv2

    try:
        data = np.fromfile(str(frame_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def parse_frame_filename(frame_path: str | Path) -> Dict[str, object]:
    """Parse a frame filename into its source video name and timestamp metadata.

    Example:
        mall_001_12.jpg -> {"video_name": "mall_001.mp4", "frame_path": "...", "timestamp_sec": 12.0}
    """

    resolved_path = Path(frame_path)
    stem_parts = resolved_path.stem.rsplit("_", 1)
    if len(stem_parts) != 2:
        raise ValueError(
            f"Invalid frame filename format: {resolved_path.name}. Expected <video_name>_<timestamp>.jpg"
        )

    video_stem, timestamp_raw = stem_parts
    try:
        timestamp_sec = float(timestamp_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp in frame filename: {resolved_path.name}"
        ) from exc

    return {
        "video_name": f"{video_stem}.mp4",
        "frame_path": str(resolved_path.resolve()),
        "timestamp_sec": timestamp_sec,
    }


def group_frames_by_video(frame_paths: List[str | Path]) -> Dict[str, List[Dict[str, object]]]:
    """Group multiple frame files by inferred source video name.

    Returns:
        {
            "mall_001.mp4": [
                {
                    "frame_path": "D:/.../mall_001_12.jpg",
                    "timestamp_sec": 12.0
                }
            ]
        }
    """

    grouped: Dict[str, List[Dict[str, object]]] = {}

    for frame_path in frame_paths:
        parsed = parse_frame_filename(frame_path)
        video_name = str(parsed["video_name"])
        grouped.setdefault(video_name, []).append(
            {
                "frame_path": str(parsed["frame_path"]),
                "timestamp_sec": float(parsed["timestamp_sec"]),
            }
        )

    for video_name, items in grouped.items():
        grouped[video_name] = sorted(items, key=lambda item: item["timestamp_sec"])

    return grouped
