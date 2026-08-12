from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image

from config import Settings
from utils.path_utils import build_asset_id, normalize_path


logger = logging.getLogger(__name__)


class AccidentDatasetImportService:
    """Import CARLA accident BEV-instance-camera sequences into ForeSea frame assets and CLIP index records.

    The dataset layout::

        type1_subtype1_accident/
        ├── ego_vehicle/
        │   └── BEV_instance_camera/
        │       ├── Town01_type001_subtype0001_scenario00004/
        │       │   ├── Town01_type001_subtype0001_scenario00004_001.npz
        │       │   ├── Town01_type001_subtype0001_scenario00004_002.npz
        │       │   └── ...
        │       └── ...
        └── meta/
            ├── Town01_type001_subtype0001_scenario00004.txt
            └── ...

    Each ``.npz`` file contains a single ``data`` array of shape ``(1200, 1200, 3)`` with
    ``uint8`` dtype representing a bird's-eye-view RGB frame.

    Each ``.txt`` meta file describes the accident scenario:

    ``ClearNoon 7341 van 7343 car 5808.36 same front 61``
    followed by key-value lines for ``colliding agents``, ``agents id``,
    ``road_type``, ``another_vehicle_spawn_side``, ``ego_vehicle_direction``,
    and ``other_vehicle_direction``.
    """

    SOURCE_MAP_FILENAME = "accident_sources.json"
    BEV_SUBDIR = Path("ego_vehicle") / "BEV_instance_camera"
    META_SUBDIR = Path("meta")
    _NPZ_FRAME_RE = re.compile(r"_(\d+)\.npz$")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.source_map_path = self.settings.dataset_metadata_dir / self.SOURCE_MAP_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        dataset_root: str | Path | None,
        scenario_names: List[str],
        max_scenarios: int | None,
        frame_step: int,
        clip_service,
        index_service,
    ) -> Dict[str, object]:
        """Import accident scenarios and push them into the CLIP index."""
        root = self._resolve_dataset_root(dataset_root)
        selected_scenarios = list(self.iter_scenarios(root, scenario_names=scenario_names))
        if max_scenarios is not None:
            selected_scenarios = selected_scenarios[:max_scenarios]

        if not selected_scenarios:
            raise FileNotFoundError(f"No accident scenarios found in {root}.")

        results: List[Dict] = []
        errors: List[str] = []
        source_map = self._load_source_map()

        logger.info(
            "Starting accident dataset import root=%s scenarios=%s frame_step=%s",
            root,
            len(selected_scenarios),
            frame_step,
        )

        for scenario_dir in selected_scenarios:
            try:
                result = self.import_scenario(
                    scenario_dir=scenario_dir,
                    dataset_root=root,
                    frame_step=frame_step,
                    clip_service=clip_service,
                    index_service=index_service,
                )
                results.append(result)
                source_map[result["video_name"]] = {
                    "video_id": result["video_id"],
                    "video_name": result["video_name"],
                    "source_scenario_dir": normalize_path(scenario_dir),
                    "meta": result.get("scenario_meta", {}),
                }
            except Exception as exc:
                logger.exception("Failed to import accident scenario: %s", scenario_dir)
                errors.append(f"{scenario_dir.name}: {exc}")

        self._save_source_map(source_map)
        return {
            "total_scenarios": len(selected_scenarios),
            "succeeded_scenarios": len(results),
            "failed_scenarios": len(errors),
            "results": results,
            "errors": errors,
        }

    def iter_scenarios(
        self,
        dataset_root: Path,
        scenario_names: List[str],
    ) -> Iterable[Path]:
        """Yield accident scenario directories under ``BEV_instance_camera/``."""
        requested_names = {name.strip() for name in scenario_names if name.strip()}
        bev_dir = dataset_root / self.BEV_SUBDIR
        if not bev_dir.exists() or not bev_dir.is_dir():
            raise FileNotFoundError(f"BEV instance camera directory not found: {bev_dir}")

        for scenario_dir in sorted(path for path in bev_dir.iterdir() if path.is_dir()):
            if requested_names and scenario_dir.name not in requested_names:
                continue
            yield scenario_dir

    def import_scenario(
        self,
        scenario_dir: Path,
        dataset_root: Path,
        frame_step: int,
        clip_service,
        index_service,
    ) -> Dict[str, object]:
        """Import one accident scenario into frame assets and CLIP index entries."""
        video_name = self._build_video_name(scenario_dir.name)
        video_id = build_asset_id(scenario_dir)
        frame_metadata = self.build_frame_metadata(
            scenario_dir=scenario_dir,
            video_name=video_name,
            frame_step=frame_step,
        )
        if not frame_metadata:
            raise ValueError(f"No frames imported from scenario: {scenario_dir}")

        imported_paths = [Path(item["frame_path"]) for item in frame_metadata]
        embeddings = clip_service.encode_image_paths(imported_paths)
        index_service.upsert_video_records(
            video_id=video_id,
            frame_metadata=frame_metadata,
            embeddings=embeddings,
        )

        scenario_meta = self.parse_meta_file(dataset_root / self.META_SUBDIR / f"{scenario_dir.name}.txt")

        logger.info(
            "Imported accident scenario=%s frames=%s video_id=%s",
            scenario_dir.name,
            len(frame_metadata),
            video_id,
        )
        return {
            "video_id": video_id,
            "video_name": video_name,
            "source_scenario_dir": normalize_path(scenario_dir),
            "imported_frames": len(frame_metadata),
            "indexed_frames": len(frame_metadata),
            "scenario_meta": scenario_meta,
        }

    def build_frame_metadata(
        self,
        scenario_dir: Path,
        video_name: str,
        frame_step: int,
    ) -> List[Dict[str, Any]]:
        """Extract BEV frames from ``.npz`` files and build CLIP frame metadata."""
        video_id = build_asset_id(scenario_dir)
        safe_video_stem = Path(video_name).stem
        metadata: List[Dict[str, Any]] = []

        npz_files = sorted(
            scenario_dir.glob("*.npz"),
            key=lambda p: self._extract_frame_index(p) or 0,
        )

        for index, source_npz in enumerate(npz_files, start=1):
            if (index - 1) % frame_step != 0:
                continue

            frame_index = self._extract_frame_index(source_npz) or (index - 1)
            timestamp_seconds = round(frame_index / self.settings.accident_frame_rate, 3)

            target_name = f"{safe_video_stem}_{timestamp_seconds:.3f}.jpg"
            target_frame = self.settings.frames_dir / target_name
            self._extract_and_save_frame(source_npz, target_frame)

            metadata.append(
                {
                    "video_id": video_id,
                    "video_name": video_name,
                    "video_path": normalize_path(scenario_dir),
                    "frame_path": normalize_path(target_frame),
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                }
            )

        return metadata

    def parse_meta_file(self, meta_path: Path) -> Dict[str, Any]:
        """Parse a CARLA accident scenario meta ``.txt`` file into structured metadata."""
        if not meta_path.exists():
            logger.warning("Meta file not found: %s", meta_path)
            return {}

        raw_lines = meta_path.read_text(encoding="utf-8").strip().splitlines()
        if not raw_lines:
            return {}

        meta: Dict[str, Any] = {}
        first_parts = raw_lines[0].split()
        if len(first_parts) >= 8:
            meta["weather"] = first_parts[0]
            meta["ego_agent_id"] = first_parts[1]
            meta["ego_agent_type"] = first_parts[2]
            meta["other_agent_id"] = first_parts[3]
            meta["other_agent_type"] = first_parts[4]
            meta["collision_distance"] = first_parts[5]
            meta["spawn_side"] = first_parts[6]
            meta["collision_position"] = first_parts[7]
            if len(first_parts) >= 9:
                meta["collision_frame"] = first_parts[8]

        for line in raw_lines[1:]:
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            meta[key.replace(" ", "_")] = value

        return meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_save_frame(self, npz_path: Path, target_frame: Path) -> Path:
        """Load a ``.npz`` BEV frame and save it as a JPEG image."""
        target_frame.parent.mkdir(parents=True, exist_ok=True)
        data = np.load(str(npz_path), allow_pickle=True)
        array = data["data"]
        if array.dtype != np.uint8:
            array = (array * 255).clip(0, 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
        Image.fromarray(array).save(str(target_frame), format="JPEG", quality=95)
        return target_frame

    def _extract_frame_index(self, npz_path: Path) -> int | None:
        match = self._NPZ_FRAME_RE.search(npz_path.name)
        if match:
            return int(match.group(1)) - 1
        return None

    def _resolve_dataset_root(self, dataset_root: str | Path | None) -> Path:
        root = (
            Path(dataset_root).expanduser().resolve()
            if dataset_root
            else self.settings.accident_dataset_dir.resolve()
        )
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Accident dataset root not found: {root}")
        return root

    def _build_video_name(self, scenario_name: str) -> str:
        return f"Accident_{scenario_name}.mp4"

    def _load_source_map(self) -> Dict[str, Dict[str, Any]]:
        if not self.source_map_path.exists():
            return {}
        try:
            payload = json.loads(self.source_map_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.exception("Failed to load accident source map: %s", self.source_map_path)
            return {}

    def _save_source_map(self, source_map: Dict[str, Dict[str, Any]]) -> None:
        self.source_map_path.write_text(
            json.dumps(source_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved accident source map to %s", self.source_map_path)
