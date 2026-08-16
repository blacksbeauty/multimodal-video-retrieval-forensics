from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import services.event_plugins  # noqa: F401 - trigger plugin registration
from config import Settings
from core.schemas import (
    DetectionVideoMetadata,
    EventMetadata,
    EventVideoMetadata,
    TrajectoryVideoMetadata,
)
from services.event_plugins.registry import get, list_plugins


logger = logging.getLogger(__name__)


class EventService:
    """Run registered event plugins over detection and trajectory metadata and persist event bundles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.event_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.settings.event_config_dir.mkdir(parents=True, exist_ok=True)

    def process_metadata_directories(
        self,
        detection_dir: str | Path | None = None,
        trajectory_dir: str | Path | None = None,
        plugin_names: List[str] | None = None,
    ) -> List[EventVideoMetadata]:
        """Generate event metadata for all trajectory bundles in a directory."""
        resolved_detection_dir = (
            Path(detection_dir).expanduser().resolve()
            if detection_dir
            else self.settings.detection_metadata_dir.resolve()
        )
        resolved_trajectory_dir = (
            Path(trajectory_dir).expanduser().resolve()
            if trajectory_dir
            else self.settings.trajectory_metadata_dir.resolve()
        )

        if not resolved_detection_dir.exists() or not resolved_detection_dir.is_dir():
            raise FileNotFoundError(f"Detection metadata directory not found: {resolved_detection_dir}")
        if not resolved_trajectory_dir.exists() or not resolved_trajectory_dir.is_dir():
            raise FileNotFoundError(f"Trajectory metadata directory not found: {resolved_trajectory_dir}")

        trajectory_paths = sorted(resolved_trajectory_dir.glob("*.json"))
        if not trajectory_paths:
            logger.warning("No trajectory metadata files found in %s", resolved_trajectory_dir)
            return []

        outputs: List[EventVideoMetadata] = []
        for trajectory_path in trajectory_paths:
            event_bundle = self.process_video_metadata(
                trajectory_path=trajectory_path,
                detection_dir=resolved_detection_dir,
                plugin_names=plugin_names,
            )
            if event_bundle is None:
                continue
            self.save_event_metadata(event_bundle)
            outputs.append(event_bundle)

        logger.info("Completed event generation videos=%s", len(outputs))
        return outputs

    def process_video_metadata(
        self,
        trajectory_path: str | Path,
        detection_dir: str | Path | None = None,
        plugin_names: List[str] | None = None,
    ) -> EventVideoMetadata | None:
        """Generate event metadata for one trajectory bundle."""
        trajectory_metadata = self._load_trajectory_metadata(Path(trajectory_path))
        if trajectory_metadata is None:
            return None

        detection_dir_path = (
            Path(detection_dir).expanduser().resolve()
            if detection_dir
            else self.settings.detection_metadata_dir.resolve()
        )
        detection_path = detection_dir_path / f"{trajectory_metadata.video_id}.json"
        detection_metadata = self._load_detection_metadata(detection_path)
        if detection_metadata is None:
            return None

        active_plugins = plugin_names or list(self.settings.event_plugin_names)
        events = self.run(
            video_id=trajectory_metadata.video_id,
            detections=detection_metadata,
            trajectories=trajectory_metadata,
            plugin_names=active_plugins,
        )
        return EventVideoMetadata(
            video_id=trajectory_metadata.video_id,
            video_name=trajectory_metadata.video_name,
            video_path=trajectory_metadata.video_path,
            events=events,
        )

    def run(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        plugin_names: List[str],
    ) -> List[EventMetadata]:
        """Execute registered plugins for one video bundle."""
        events: List[EventMetadata] = []
        for plugin_name in plugin_names:
            plugin_cls = get(plugin_name)
            if plugin_cls is None:
                logger.warning("Event plugin not registered: %s. Available=%s", plugin_name, list_plugins())
                continue

            plugin = plugin_cls()
            config = self._load_plugin_config(plugin_name)
            try:
                produced = plugin.execute(
                    video_id=video_id,
                    detections=detections,
                    trajectories=trajectories,
                    config=config,
                )
            except Exception:
                # Code Review Must Fix #7：单个插件异常只跳过该插件，不中断整批事件生成。
                logger.exception(
                    "Event plugin %s failed for video_id=%s; skipping plugin",
                    plugin_name,
                    video_id,
                )
                continue
            events.extend(produced)

        events.sort(key=lambda item: (item.start_ts, item.end_ts, item.event_type, item.event_id))
        return events

    def save_event_metadata(self, metadata: EventVideoMetadata) -> Path:
        """Persist event metadata under metadata/events/<video_id>.json."""
        output_path = self.settings.event_metadata_dir / f"{metadata.video_id}.json"
        payload = metadata.model_dump_json(indent=2)
        temp_output_path = output_path.with_name(f"{output_path.name}.{time.time_ns()}.tmp")
        temp_output_path.write_text(payload, encoding="utf-8")
        temp_output_path.replace(output_path)
        logger.info("Saved event metadata to %s", output_path)
        return output_path

    def _load_detection_metadata(self, metadata_path: Path) -> DetectionVideoMetadata | None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return DetectionVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load detection metadata for events: %s", metadata_path)
            return None

    def _load_trajectory_metadata(self, metadata_path: Path) -> TrajectoryVideoMetadata | None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return TrajectoryVideoMetadata.model_validate(payload)
        except Exception:
            logger.exception("Failed to load trajectory metadata for events: %s", metadata_path)
            return None

    def _load_plugin_config(self, plugin_name: str) -> Dict[str, object]:
        config_path = self.settings.event_config_dir / f"{plugin_name}.json"
        if not config_path.exists():
            return {}
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.exception("Failed to load event plugin config: %s", config_path)
            return {}
