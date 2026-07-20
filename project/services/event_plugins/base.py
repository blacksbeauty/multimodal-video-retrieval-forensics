from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.schemas import DetectionVideoMetadata, EventMetadata, TrajectoryVideoMetadata


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
