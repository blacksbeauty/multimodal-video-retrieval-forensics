from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from config import Settings


logger = logging.getLogger(__name__)


class SearchService:
    """CLIP 语义检索服务（纯文本→向量→FAISS 相似度检索）。

    通道状态约定：CLIP 模型懒加载；模型不可用或 FAISS 未建索引时
    返回空列表而非抛 500（P5 降级），由上层混合检索继续走其他通道。
    """

    def __init__(self, settings: Settings, embedding_service, index_service) -> None:
        self.settings = settings
        self.embedding_service = embedding_service  # 文本/图像编码器（CN-CLIP 或 OpenCLIP）
        self.index_service = index_service          # FAISS 索引访问层

    def search_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """单查询文本的 CLIP 语义检索（帧/段级结果，按分数降序）。

        返回值含 S7 段级字段透传（segment_id/segment_text/time_range/start_ts/end_ts），
        帧级结果这些字段为空，保证下游融合逻辑统一处理。
        注意：use_segment_index=False 时丢弃段级记录，仅保留帧级（一键回滚开关）。
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        # P5 degradation: if the CLIP/FAISS channels are disabled or the model
        # failed to load, degrade to an empty semantic result instead of 500.
        if not self.settings.enable_clip or not self.settings.enable_faiss:
            logger.warning("CLIP/FAISS channel disabled; returning empty semantic results")
            return []
        self.embedding_service.load_model()
        if not self.embedding_service.is_available():
            logger.warning("CLIP model unavailable; returning empty semantic results")
            return []

        faiss_service = self.index_service.faiss_service
        if faiss_service.index is None or not faiss_service.metadata:
            logger.warning("Search requested before any embeddings were indexed.")
            return []

        query_embedding = self.embedding_service.encode_text(query)
        raw_results = faiss_service.search(query_embedding=query_embedding, top_k=top_k)

        results = []
        for item in raw_results:
            segment_id = item.get("segment_id", "")
            if segment_id and not self.settings.use_segment_index:
                # S7: 回滚开关——use_segment_index=False 时丢弃段级记录，仅帧级检索
                continue
            results.append(
                {
                    "video_id": item.get("video_id", ""),
                    "video_name": item.get("video_name", ""),
                    "video_path": item.get("video_path", ""),
                    "timestamp": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                    "timestamp_seconds": float(item.get("timestamp_seconds", item.get("timestamp", 0.0))),
                    "score": float(item.get("score", 0.0)),
                    "frame_path": item.get("frame_path", ""),
                    "frame_id": item.get("frame_id", ""),
                    # S7: 段级字段透传（无则空，保持帧级兼容）
                    "segment_id": segment_id,
                    "segment_text": item.get("segment_text", item.get("text", "")),
                    "time_range": item.get("time_range", {}),
                    "start_ts": float(item.get("start_ts", 0.0)),
                    "end_ts": float(item.get("end_ts", 0.0)),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        logger.info(
            "Text search completed query=%s top_k=%s results=%s",
            query,
            top_k,
            len(results),
        )
        return results

    def search_text_variants(self, queries: Iterable[str], top_k: int = 5) -> List[Dict]:
        """多查询变体检索并合并去重（混合检索的 CLIP 通道入口）。

        对每个改写查询变体分别做 top_k×multiplier 候选检索，再按
        segment_id / frame_id / frame_path 去重合并，保留各 key 下最高分记录，
        最终按分数降序返回（hybrid 候选量由 hybrid_candidate_multiplier 放大）。
        """
        normalized_queries = [query.strip() for query in queries if query and query.strip()]
        if not normalized_queries:
            raise ValueError("At least one non-empty query variant is required.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        merged: Dict[str, Dict] = {}
        candidate_top_k = max(top_k, top_k * self.settings.hybrid_candidate_multiplier)

        for query in normalized_queries:
            for item in self.search_text(query=query, top_k=candidate_top_k):
                # S7: 段记录用 segment_id 作 key，帧记录用 frame_id/frame_path
                frame_key = str(
                    item.get("segment_id")
                    or item.get("frame_id")
                    or item.get("frame_path")
                )
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
