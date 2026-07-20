from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from config import Settings
from utils.path_utils import build_asset_id, normalize_path


logger = logging.getLogger(__name__)


class StreetSceneImportService:
    """Import StreetScene image sequences into ForeSea-standard frame assets and CLIP index records."""

    SOURCE_MAP_FILENAME = "streetscene_sources.json"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.source_map_path = self.settings.dataset_metadata_dir / self.SOURCE_MAP_FILENAME

    def ingest_directory(
        self,
        dataset_root: str | Path | None,
        split: str | None,
        sequence_names: List[str],
        max_sequences: int | None,
        frame_step: int,
        clip_service,
        index_service,
    ) -> Dict[str, object]:
        """Import StreetScene sequences and push them into the CLIP index."""
        root = self._resolve_dataset_root(dataset_root)
        selected_sequences = list(self.iter_sequences(root, split=split, sequence_names=sequence_names))
        if max_sequences is not None:
            selected_sequences = selected_sequences[:max_sequences]

        if not selected_sequences:
            raise FileNotFoundError(f"No StreetScene sequences found in {root} for split={split!r}.")

        results: List[Dict] = []
        errors: List[str] = []
        source_map = self._load_source_map()

        logger.info(
            "Starting StreetScene import root=%s split=%s sequences=%s frame_step=%s",
            root,
            split,
            len(selected_sequences),
            frame_step,
        )

        for split_name, sequence_dir in selected_sequences:
            try:
                result = self.import_sequence(
                    sequence_dir=sequence_dir,
                    split_name=split_name,
                    frame_step=frame_step,
                    clip_service=clip_service,
                    index_service=index_service,
                )
                results.append(result)
                source_map[result["video_name"]] = {
                    "video_id": result["video_id"],
                    "video_name": result["video_name"],
                    "source_sequence_dir": normalize_path(sequence_dir),
                }
            except Exception as exc:
                logger.exception("Failed to import StreetScene sequence: %s", sequence_dir)
                errors.append(f"{sequence_dir.name}: {exc}")

        self._save_source_map(source_map)
        return {
            "total_sequences": len(selected_sequences),
            "succeeded_sequences": len(results),
            "failed_sequences": len(errors),
            "results": results,
            "errors": errors,
        }

    def iter_sequences(
        self,
        dataset_root: Path,
        split: str | None,
        sequence_names: List[str],
    ) -> Iterable[tuple[str, Path]]:
        """Yield StreetScene sequence directories for the requested split or names."""
        requested_names = {name.strip() for name in sequence_names if name.strip()}

        split_candidates = []
        if split and split.lower() not in {"all", "both"}:
            split_candidates = [split]
        else:
            split_candidates = ["Train", "Test"]

        for split_name in split_candidates:
            split_dir = dataset_root / split_name
            if not split_dir.exists() or not split_dir.is_dir():
                continue

            for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                if requested_names and sequence_dir.name not in requested_names:
                    continue
                yield split_name, sequence_dir

    def import_sequence(
        self,
        sequence_dir: Path,
        split_name: str,
        frame_step: int,
        clip_service,
        index_service,
    ) -> Dict[str, object]:
        """Import one StreetScene sequence into frame assets and CLIP index entries."""
        video_name = self._build_video_name(split_name, sequence_dir.name)
        video_id = build_asset_id(sequence_dir)
        frame_metadata = self.build_frame_metadata(sequence_dir, video_name=video_name, frame_step=frame_step)
        if not frame_metadata:
            raise ValueError(f"No frames imported from sequence: {sequence_dir}")

        imported_paths = [Path(item["frame_path"]) for item in frame_metadata]
        embeddings = clip_service.encode_image_paths(imported_paths)
        index_service.upsert_video_records(video_id=video_id, frame_metadata=frame_metadata, embeddings=embeddings)

        logger.info(
            "Imported StreetScene sequence=%s split=%s frames=%s video_id=%s",
            sequence_dir.name,
            split_name,
            len(frame_metadata),
            video_id,
        )
        return {
            "video_id": video_id,
            "video_name": video_name,
            "source_sequence_dir": normalize_path(sequence_dir),
            "imported_frames": len(frame_metadata),
            "indexed_frames": len(frame_metadata),
        }

    def build_frame_metadata(self, sequence_dir: Path, video_name: str, frame_step: int) -> List[Dict]:
        """Copy StreetScene frames into ForeSea assets and build CLIP frame metadata."""
        video_id = build_asset_id(sequence_dir)
        safe_video_stem = Path(video_name).stem
        metadata: List[Dict] = []

        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        for index, source_frame in enumerate(frame_paths, start=1):
            if (index - 1) % frame_step != 0:
                continue

            timestamp_seconds = self._frame_timestamp(source_frame)
            target_name = f"{safe_video_stem}_{timestamp_seconds:.3f}.jpg"
            target_frame = self.settings.frames_dir / target_name
            self.copy_or_link_frame(source_frame, target_frame)

            metadata.append(
                {
                    "video_id": video_id,
                    "video_name": video_name,
                    "video_path": normalize_path(sequence_dir),
                    "frame_path": normalize_path(target_frame),
                    "frame_index": self._frame_index(source_frame),
                    "timestamp_seconds": timestamp_seconds,
                }
            )

        return metadata

    def copy_or_link_frame(self, source_frame: Path, target_frame: Path) -> Path:
        """Copy a StreetScene frame into the managed ForeSea frames directory."""
        target_frame.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_frame, target_frame)
        return target_frame

    def _resolve_dataset_root(self, dataset_root: str | Path | None) -> Path:
        root = Path(dataset_root).expanduser().resolve() if dataset_root else self.settings.streetscene_dataset_dir.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"StreetScene dataset root not found: {root}")
        return root

    def _build_video_name(self, split_name: str, sequence_name: str) -> str:
        return f"StreetScene_{split_name}_{sequence_name}.mp4"

    def _frame_index(self, frame_path: Path) -> int:
        try:
            return int(frame_path.stem) - 1
        except ValueError:
            return 0

    def _frame_timestamp(self, frame_path: Path) -> float:
        return round(self._frame_index(frame_path) / self.settings.streetscene_frame_rate, 3)

    def _load_source_map(self) -> Dict[str, Dict[str, str]]:
        if not self.source_map_path.exists():
            return {}
        try:
            payload = json.loads(self.source_map_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.exception("Failed to load StreetScene source map: %s", self.source_map_path)
            return {}

    def _save_source_map(self, source_map: Dict[str, Dict[str, str]]) -> None:
        self.source_map_path.write_text(
            json.dumps(source_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved StreetScene source map to %s", self.source_map_path)
