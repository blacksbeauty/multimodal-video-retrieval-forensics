from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np

from config import Settings


logger = logging.getLogger(__name__)


class FaissIndexService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._faiss = None
        self.index = None
        self.metadata: List[dict] = []

    def build_index(self, embeddings: np.ndarray, metadata: Sequence[dict] | None = None):
        matrix = self._normalize_embeddings(embeddings)
        normalized_metadata = list(metadata or [])
        if normalized_metadata and len(normalized_metadata) != matrix.shape[0]:
            raise ValueError("Metadata length must match embedding count.")

        faiss = self._get_faiss()

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self.index = index
        self.metadata = normalized_metadata

        logger.info(
            "Built FAISS index with vectors=%s dim=%s",
            matrix.shape[0],
            matrix.shape[1],
        )
        return self.index

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")
        if self.index.ntotal == 0:
            return []

        query = self._normalize_query(query_embedding)
        search_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, search_k)

        results: List[dict] = []
        for rank, (score, item_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if item_index < 0:
                continue

            base = self.metadata[item_index] if item_index < len(self.metadata) else {}
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": int(item_index),
                    **base,
                }
            )

        logger.info("FAISS search completed with top_k=%s hits=%s", top_k, len(results))
        return results

    def save_index(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> dict:
        if self.index is None:
            raise ValueError("No FAISS index available to save.")

        faiss = self._get_faiss()
        resolved_index_path = Path(index_path or self.settings.faiss_index_path)
        resolved_metadata_path = Path(metadata_path or self.settings.metadata_path)

        resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        serialized_index = faiss.serialize_index(self.index)
        resolved_index_path.write_bytes(serialized_index.tobytes())
        resolved_metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved FAISS index to %s", resolved_index_path)
        return {
            "index_path": str(resolved_index_path),
            "metadata_path": str(resolved_metadata_path),
            "count": int(self.index.ntotal),
        }

    def load_index(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        resolved_index_path = Path(index_path or self.settings.faiss_index_path)
        resolved_metadata_path = Path(metadata_path or self.settings.metadata_path)

        if not resolved_index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {resolved_index_path}")
        if not resolved_metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata file not found: {resolved_metadata_path}")

        faiss = self._get_faiss()
        serialized_index = np.frombuffer(resolved_index_path.read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(serialized_index)
        self.metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))

        logger.info(
            "Loaded FAISS index from %s with vectors=%s",
            resolved_index_path,
            self.index.ntotal,
        )
        return self.index

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
