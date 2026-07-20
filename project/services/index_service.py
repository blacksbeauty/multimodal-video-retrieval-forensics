from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import Settings
from services.faiss_index_service import FaissIndexService
from utils.path_utils import build_asset_id, build_frame_id, normalize_path


logger = logging.getLogger(__name__)


class IndexService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.faiss_service = FaissIndexService(settings)
        self._load_index_from_disk()

    def _embedding_file(self, video_id: str) -> Path:
        return self.settings.embeddings_dir / f"{video_id}.npy"

    def _metadata_file(self, video_id: str) -> Path:
        return self.settings.embeddings_dir / f"{video_id}.json"

    def _load_index_from_disk(self) -> None:
        if (
            not self.settings.faiss_index_path.exists()
            or not self.settings.metadata_path.exists()
        ):
            return

        self.faiss_service.load_index()
        if self._index_needs_rebuild(self.faiss_service.metadata):
            logger.info("Loaded index metadata is stale. Rebuilding FAISS index from embedding bundles.")
            self.rebuild_index()
            return

        logger.info(
            "Loaded existing FAISS index with %s frame vectors.",
            len(self.faiss_service.metadata),
        )

    def upsert_video_records(
        self,
        video_id: str,
        frame_metadata: List[Dict],
        embeddings: np.ndarray,
    ) -> None:
        if len(frame_metadata) != len(embeddings):
            raise ValueError("Frame metadata count does not match embedding count.")

        normalized_metadata = [self._normalize_frame_metadata(item) for item in frame_metadata]
        np.save(self._embedding_file(video_id), embeddings.astype(np.float32))
        self._metadata_file(video_id).write_text(
            json.dumps(normalized_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %s embeddings for video_id=%s", len(normalized_metadata), video_id)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        embedding_files = sorted(self.settings.embeddings_dir.glob("*.npy"))
        valid_embeddings: List[np.ndarray] = []
        valid_metadata: List[Dict] = []
        seen_frame_ids: set[str] = set()

        total_frames = 0
        duplicate_frames_skipped = 0
        invalid_embeddings_skipped = 0

        for embedding_file in embedding_files:
            bundle = self._load_embedding_bundle(embedding_file)
            if bundle is None:
                continue

            embeddings, metadata = bundle
            total_frames += len(metadata)

            for embedding, raw_metadata in zip(embeddings, metadata):
                try:
                    normalized_metadata = self._normalize_frame_metadata(raw_metadata)
                except (KeyError, TypeError, ValueError):
                    invalid_embeddings_skipped += 1
                    logger.warning(
                        "Skipping invalid frame metadata from bundle=%s",
                        embedding_file.stem,
                    )
                    continue

                frame_id = normalized_metadata["frame_id"]
                if frame_id in seen_frame_ids:
                    duplicate_frames_skipped += 1
                    continue

                seen_frame_ids.add(frame_id)
                valid_embeddings.append(np.asarray(embedding, dtype=np.float32))
                valid_metadata.append(normalized_metadata)

        if not valid_embeddings:
            self.faiss_service.index = None
            self.faiss_service.metadata = []
            if self.settings.faiss_index_path.exists():
                self.settings.faiss_index_path.unlink()
            self.settings.metadata_path.write_text("[]", encoding="utf-8")
            logger.info(
                "No valid embeddings found. Cleared FAISS index. total_frames=%s duplicate_frames_skipped=%s invalid_embeddings_skipped=%s",
                total_frames,
                duplicate_frames_skipped,
                invalid_embeddings_skipped,
            )
            return

        matrix = np.vstack(valid_embeddings).astype(np.float32)
        self.faiss_service.build_index(matrix, valid_metadata)
        self.faiss_service.save_index()
        logger.info(
            "Rebuilt FAISS index total_frames=%s duplicate_frames_skipped=%s invalid_embeddings_skipped=%s rebuilt_embedding_count=%s",
            total_frames,
            duplicate_frames_skipped,
            invalid_embeddings_skipped,
            len(self.faiss_service.metadata),
        )

    def search_text(self, query: str, top_k: int, clip_service) -> List[Dict]:
        if self.faiss_service.index is None or not self.faiss_service.metadata:
            return []

        query_vector = clip_service.encode_text(query)
        return self.faiss_service.search(query_vector, top_k)

    def get_stats(self) -> Dict[str, object]:
        metadata = self.faiss_service.metadata
        video_ids = {item["video_id"] for item in metadata} if metadata else set()
        return {
            "indexed_frames": len(metadata),
            "indexed_videos": len(video_ids),
            "index_path": str(self.settings.faiss_index_path),
            "metadata_path": str(self.settings.metadata_path),
        }

    def _load_embedding_bundle(self, embedding_file: Path) -> tuple[np.ndarray, List[Dict]] | None:
        metadata_file = embedding_file.with_suffix(".json")
        if not metadata_file.exists():
            logger.warning("Skipping embedding file without metadata: %s", embedding_file)
            return None

        try:
            embeddings = np.load(embedding_file).astype(np.float32)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load embedding bundle: %s", embedding_file.stem)
            return None

        if not isinstance(metadata, list):
            logger.warning("Skipping invalid metadata payload: %s", metadata_file)
            return None

        if len(metadata) != len(embeddings):
            logger.warning("Skipping mismatched bundle: %s", embedding_file.stem)
            return None

        return embeddings, metadata

    def _normalize_frame_metadata(self, metadata: Dict) -> Dict:
        raw_video_path = metadata.get("video_path") or metadata.get("source_video_path")
        raw_frame_path = metadata.get("frame_path")
        if not raw_video_path or not raw_frame_path:
            raise KeyError("Frame metadata must include video_path and frame_path.")

        if not Path(raw_video_path).expanduser().exists():
            raise ValueError(f"Video path does not exist: {raw_video_path}")
        if not Path(raw_frame_path).expanduser().exists():
            raise ValueError(f"Frame path does not exist: {raw_frame_path}")

        timestamp_seconds = float(metadata.get("timestamp_seconds", metadata.get("timestamp", 0.0)))
        normalized_video_path = normalize_path(raw_video_path)
        normalized_frame_path = normalize_path(raw_frame_path)
        video_path_obj = Path(normalized_video_path)
        video_id = build_asset_id(Path(raw_video_path))
        frame_id = build_frame_id(normalized_video_path, timestamp_seconds)

        return {
            **metadata,
            "video_id": str(video_id),
            "frame_id": frame_id,
            "video_name": str(metadata.get("video_name") or video_path_obj.name),
            "video_path": normalized_video_path,
            "frame_path": normalized_frame_path,
            "timestamp_seconds": timestamp_seconds,
        }

    def _index_needs_rebuild(self, metadata: List[Dict]) -> bool:
        """Check whether persisted index metadata is missing normalized fields."""
        if not metadata:
            return False

        sample = metadata[0]
        video_path = str(sample.get("video_path", ""))
        frame_path = str(sample.get("frame_path", ""))
        frame_id = str(sample.get("frame_id", ""))
        if not frame_id:
            return True
        if video_path != video_path.lower() or frame_path != frame_path.lower():
            return True
        if "\\" in video_path or "\\" in frame_path:
            return True
        return False
