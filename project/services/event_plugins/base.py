from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.schemas import DetectionVideoMetadata, EventMetadata, TrajectoryVideoMetadata

# 检测类别 → 中文（用于事件 description / segment.text，数据规范 v1.1 §3.7/§3.8）
LABEL_CN: Dict[str, str] = {
    "car": "小汽车",
    "truck": "货车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "person": "行人",
    "traffic light": "信号灯",
    "bicycle": "自行车",
    "van": "面包车",
}


def label_to_chinese(label: str) -> str:
    """S7: 检测类别英文 label → 中文（未知类别回退原文）。"""
    return LABEL_CN.get(str(label).strip().casefold(), str(label))


class EventPluginBase(ABC):
    """Abstract base class for all rule-based traffic event plugins."""

    plugin_name: str = ""
    event_type: str = ""

    @abstractmethod
    def execute(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        config: Dict[str, Any],
    ) -> List[EventMetadata]:
        """Produce traffic events for one video."""
