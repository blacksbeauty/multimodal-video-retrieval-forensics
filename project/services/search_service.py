from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from config import Settings


logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, settings: Settings, embedding_service, index_service) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.index_service = index_service

    def search_text(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        faiss_service = self.index_service.faiss_service
        if faiss_service.index is None or not faiss_service.metadata:
            logger.warning("Search requested before any embeddings were indexed.")
            return []

        query_embedding = self.embedding_service.encode_text(query)
        raw_results = faiss_service.search(query_embedding=query_embedding, top_k=top_k)

        results = [
            {
                "video_id": item.get("video_id", ""),
                "video_name": item.get("video_name", ""),
                "video_path": item.get("video_path", ""),
                "timestamp": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                "timestamp_seconds": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                "score": float(item.get("score", 0.0)),
                "frame_path": item.get("frame_path", ""),
                "frame_id": item.get("frame_id", ""),
            }
            for item in raw_results
        ]

        results.sort(key=lambda item: item["score"], reverse=True)
        logger.info(
            "Text search completed query=%s top_k=%s results=%s",
            query,
            top_k,
            len(results),
        )
        return results

    def search_text_variants(self, queries: Iterable[str], top_k: int = 5) -> List[Dict]:
        """Search CLIP with multiple query variants and merge the best frame-level hits."""
        normalized_queries = [query.strip() for query in queries if query and query.strip()]
        if not normalized_queries:
            raise ValueError("At least one non-empty query variant is required.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        merged: Dict[str, Dict] = {}
        candidate_top_k = max(top_k, top_k * self.settings.hybrid_candidate_multiplier)

        for query in normalized_queries:
            for item in self.search_text(query=query, top_k=candidate_top_k):
                frame_key = str(item.get("frame_id") or item.get("frame_path"))
                enriched = {
                    **item,
                    "matched_by": ["clip"],
                    "matched_query": query,
                }
                existing = merged.get(frame_key)
                if existing is None or enriched["score"] > existing["score"]:
                    merged[frame_key] = enriched

        results = sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        logger.info(
            "Variant CLIP search completed variants=%s merged_results=%s",
            len(normalized_queries),
            len(results),
        )
        return results[:candidate_top_k]
