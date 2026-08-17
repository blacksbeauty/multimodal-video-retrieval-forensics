from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from config import Settings


logger = logging.getLogger(__name__)


class FaissIndexService:
    """FAISS 向量索引服务（**增量式** 视频级 upsert）。

    设计要点：
      - 使用 ``faiss.IndexIDMap`` + ``add_with_ids``：新增视频只追加向量，
        不重建整个索引（避免大库全量重建的 O(N) 开销）；
      - 持久化 ``video_id -> [外部id]`` 映射：支持按视频覆盖（先 remove 旧 id 再 add）
        或整体删除；
      - 全局递增 ``_next_id`` 保证所有视频的向量 id 全局唯一，避免删除后 id 复用冲突。

    持久化文件布局：
      * ``index/video_frames.index``  — 序列化后的 IndexIDMap（faiss.write_index）
      * ``index/segment_meta.json``   — {外部id: 段/帧元数据}（S6 起由 frame_metadata.json 更名）
      * ``index/video_mapping.json``  — {video_id: {"ids": [...], "count": N}}
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._faiss = None
        self.index = None  # IndexIDMap | None
        self.metadata: Dict[int, dict] = {}  # external_id -> frame metadata
        self.mapping: Dict[str, dict] = {}  # video_id -> {"ids": [...], "count": N}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    # Build / upsert / remove
    # ------------------------------------------------------------------ #

    def _create_index(self, dim: int):
        faiss = self._get_faiss()
        base = faiss.IndexFlatIP(int(dim))
        return faiss.IndexIDMap(base)

    def build_index(self, embeddings: np.ndarray, metadata: Sequence[dict] | None = None):
        """Full rebuild from a matrix (used on first build or format migration)."""
        matrix = self._normalize_embeddings(embeddings)
        meta_list = list(metadata or [])
        if meta_list and len(meta_list) != matrix.shape[0]:
            raise ValueError("Metadata length must match embedding count.")

        index = self._create_index(matrix.shape[1])
        ids = np.arange(matrix.shape[0], dtype=np.int64)
        index.add_with_ids(matrix, ids)
        self.index = index

        self.metadata = {int(i): dict(item) for i, item in enumerate(meta_list)} if meta_list else {}
        self.mapping = {}
        for i, item in enumerate(meta_list):
            video_id = str(item.get("video_id", ""))
            if video_id:
                self.mapping.setdefault(video_id, {"ids": [], "count": 0})["ids"].append(int(i))
        for entry in self.mapping.values():
            entry["count"] = len(entry["ids"])
        self._next_id = len(meta_list)

        logger.info(
            "Built FAISS index vectors=%s dim=%s videos=%s (incremental mode)",
            matrix.shape[0],
            matrix.shape[1],
            len(self.mapping),
        )
        return self.index

    def add_video(
        self,
        video_id: str,
        embeddings: np.ndarray,
        metadata: Sequence[dict],
        append: bool = False,
    ) -> dict:
        """增量新增（或覆盖）单个视频的全部向量。

        覆盖语义（append=False，默认）：若 video_id 已存在，先移除其旧 id
        （IndexIDMap.remove_ids），再追加新向量——索引其余部分不受影响。

        追加语义（append=True）：不删除该视频已有 id，仅分配新 id 追加，
        mapping 条目合并（用于同一 video_id 的帧级+段级两批向量共存；
        Code Review Must Fix：upsert_segment_records 曾误用覆盖语义
        导致段级写入删光同视频的帧级向量）。

        约束：向量维度必须与索引一致（不一致抛 ValueError）；空向量拒绝写入。
        """
        matrix = self._normalize_embeddings(embeddings)
        meta_list = list(metadata or [])
        if len(meta_list) != matrix.shape[0]:
            raise ValueError("Metadata count must match embedding count.")
        if matrix.shape[0] == 0:
            raise ValueError("Cannot add empty embeddings.")

        if self.index is None:
            self.index = self._create_index(matrix.shape[1])
        if self.index.d != matrix.shape[1]:
            raise ValueError(
                f"Embedding dim {matrix.shape[1]} does not match index dim {self.index.d}."
            )

        # 覆盖模式：先删除该 video_id 的旧 id；追加模式保留（帧级+段级共存）。
        existing = self.mapping.get(video_id)
        if existing and existing.get("ids") and not append:
            self.index.remove_ids(np.asarray(existing["ids"], dtype=np.int64))
            for old_id in existing["ids"]:
                self.metadata.pop(int(old_id), None)

        # Allocate globally increasing ids (unique across all videos).
        count = matrix.shape[0]
        ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)
        self._next_id += count

        self.index.add_with_ids(matrix, ids)
        for external_id, item in zip(ids.tolist(), meta_list):
            self.metadata[int(external_id)] = dict(item)

        if append and existing and existing.get("ids"):
            existing["ids"].extend(ids.tolist())
            existing["count"] = len(existing["ids"])
        else:
            self.mapping[video_id] = {"ids": ids.tolist(), "count": count}

        logger.info(
            "Incrementally added video_id=%s vectors=%s total=%s append=%s",
            video_id,
            count,
            self.index.ntotal,
            append,
        )
        return {"video_id": video_id, "added": count}

    def remove_video(self, video_id: str) -> bool:
        """Remove all vectors of a video (optional API)."""
        if video_id not in self.mapping:
            return False
        ids = self.mapping[video_id].get("ids") or []
        if ids and self.index is not None:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))
        for external_id in ids:
            self.metadata.pop(int(external_id), None)
        del self.mapping[video_id]
        logger.info("Removed video_id=%s vectors=%s", video_id, len(ids))
        return True

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")
        if self.index.ntotal == 0:
            return []

        query = self._normalize_query(query_embedding)
        search_k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(query, search_k)

        results: List[dict] = []
        for rank, (score, external_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if external_id < 0:
                continue
            base = self.metadata.get(int(external_id), {})
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(external_id),
                    **base,
                }
            )

        logger.info("FAISS search completed top_k=%s hits=%s", top_k, len(results))
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_index(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
    ) -> dict:
        if self.index is None:
            raise ValueError("No FAISS index available to save.")

        faiss = self._get_faiss()
        resolved_index_path = Path(index_path or self.settings.faiss_index_path)
        resolved_metadata_path = Path(metadata_path or self.settings.metadata_path)
        resolved_mapping_path = Path(mapping_path or self.settings.index_dir / "video_mapping.json")

        resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_mapping_path.parent.mkdir(parents=True, exist_ok=True)

        # tmp 文件（同一目录保证 os.replace 原子性）
        tmp_index_path = resolved_index_path.with_name(f"{resolved_index_path.name}.{os.getpid()}.tmp")
        tmp_metadata_path = resolved_metadata_path.with_name(f"{resolved_metadata_path.name}.{os.getpid()}.tmp")
        tmp_mapping_path = resolved_mapping_path.with_name(f"{resolved_mapping_path.name}.{os.getpid()}.tmp")

        faiss.write_index(self.index, str(tmp_index_path))
        tmp_metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_mapping_path.write_text(
            json.dumps(self.mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 原子替换（Code Review P1-2）：先写完整 tmp 再 os.replace，
        # 避免进程崩溃时留下 三个文件新旧不一致 的中间态。
        os.replace(tmp_index_path, resolved_index_path)
        os.replace(tmp_metadata_path, resolved_metadata_path)
        os.replace(tmp_mapping_path, resolved_mapping_path)
        logger.info(
            "Saved FAISS index vectors=%s videos=%s",
            self.index.ntotal,
            len(self.mapping),
        )
        return {
            "index_path": str(resolved_index_path),
            "metadata_path": str(resolved_metadata_path),
            "mapping_path": str(resolved_mapping_path),
            "count": int(self.index.ntotal),
        }

    def load_index(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
    ):
        """从磁盘加载索引 + 元数据 + 映射。

        兼容性处理：
          - S6 旧文件名：segment_meta.json 不存在时回退读 frame_metadata.json；
          - 旧版 metadata 若为 list（无外部 id），自动转换为 {i: meta} 字典；
          - 旧版裸 IndexFlatIP（非 IndexIDMap）无法增量 upsert，抛 TypeError
            提示需要重建索引。
        加载后 _next_id 取 max(metadata keys)+1，保证后续新增 id 不与已有 id 冲突。
        """
        resolved_index_path = Path(index_path or self.settings.faiss_index_path)
        resolved_metadata_path = Path(metadata_path or self.settings.metadata_path)
        resolved_mapping_path = Path(mapping_path or self.settings.index_dir / "video_mapping.json")

        # S6: 旧文件名兼容。新名 segment_meta.json 不存在时，回退读取旧 frame_metadata.json，
        # 首次 save_index 时自动迁移到新名。
        legacy_metadata_path = self.settings.index_dir / "frame_metadata.json"
        if not resolved_metadata_path.exists() and legacy_metadata_path.exists():
            logger.info(
                "segment_meta.json not found; falling back to legacy frame_metadata.json (%s)",
                legacy_metadata_path,
            )
            resolved_metadata_path = legacy_metadata_path

        if not resolved_index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {resolved_index_path}")
        if not resolved_metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata file not found: {resolved_metadata_path}")

        faiss = self._get_faiss()
        self.index = faiss.read_index(str(resolved_index_path))
        # Legacy indexes were plain IndexFlatIP without external ids; only the
        # new IndexIDMap format is directly usable for incremental upserts.
        if not isinstance(self.index, faiss.IndexIDMap):
            raise TypeError(
                "Index is not in incremental (IndexIDMap) format; a rebuild is required."
            )

        payload = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            # Legacy metadata list → convert to {external_id: meta}.
            self.metadata = {i: dict(item) for i, item in enumerate(payload)}
        elif isinstance(payload, dict):
            self.metadata = {int(key): dict(value) for key, value in payload.items()}
        else:
            raise TypeError(f"Unexpected metadata payload type: {type(payload).__name__}")

        if resolved_mapping_path.exists():
            raw_mapping = json.loads(resolved_mapping_path.read_text(encoding="utf-8"))
            self.mapping = {
                str(video_id): {"ids": [int(i) for i in entry.get("ids", [])], "count": int(entry.get("count", len(entry.get("ids", []))))}
                for video_id, entry in raw_mapping.items()
            }
        else:
            self.mapping = {}

        self._next_id = (max(self.metadata.keys()) + 1) if self.metadata else 0

        logger.info(
            "Loaded FAISS index vectors=%s videos=%s (incremental mode)",
            self.index.ntotal,
            len(self.mapping),
        )
        return self.index

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_faiss(self):
        if self._faiss is None:
            import faiss

            self._faiss = faiss
        return self._faiss

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array.")
        if matrix.shape[0] == 0:
            raise ValueError("embeddings cannot be empty.")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _normalize_query(self, query_embedding: np.ndarray) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_embedding must be a 1D vector or shape (1, dim).")

        norms = np.linalg.norm(query, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return query / norms
