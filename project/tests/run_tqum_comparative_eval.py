#!/usr/bin/env python
"""
CN-CLIP 中文语义泛化评测脚本 V2 — TQUM 对比版

对比两种模式：
  Mode A (baseline): 纯 CN-CLIP 文本编码 → FAISS 检索（上一轮基线）
  Mode B (TQUM):     TQUM 语义解析 → 事件路由 + 中文查询扩展 → 多通道检索

Mode B 的检索策略：
  - event 类型查询 → 搜索事件元数据（按事件类型匹配，按置信度排序）
  - object/attribute 类型查询 → TQUM 中文扩展 → CN-CLIP 编码 → FAISS 检索
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from services.embedding_service import EmbeddingService
from services.query_rewrite_service import QueryRewriteService
from services.hybrid_search_service import HybridSearchService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 视频标签映射
# ---------------------------------------------------------------------------
WRONG_WAY = ["MVI_40854", "MVI_40903"]
RED_LIGHT = ["MVI_40853"]
CROSS_LINE = ["MVI_40852"]
WHITE_CAR = ["MVI_40855"]
EVENT_VIDEOS = ["MVI_40852", "MVI_40853", "MVI_40854", "MVI_40903"]

# ---------------------------------------------------------------------------
# 100 条测试集（与 V1 完全相同）
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Dict] = [
    # ==================== 1. 标准描述 (20) ====================
    {"query": "红色汽车逆行", "expected": ["MVI_40854"], "category": "标准描述"},
    {"query": "机动车逆向行驶", "expected": ["MVI_40903"], "category": "标准描述"},
    {"query": "车辆闯红灯", "expected": ["MVI_40853"], "category": "标准描述"},
    {"query": "货车压线", "expected": ["MVI_40852"], "category": "标准描述"},
    {"query": "白色汽车", "expected": ["MVI_40855"], "category": "标准描述"},
    {"query": "红色轿车逆向驾驶", "expected": ["MVI_40854"], "category": "标准描述"},
    {"query": "车辆方向与道路规定相反", "expected": ["MVI_40854"], "category": "标准描述"},
    {"query": "汽车违反交通信号灯", "expected": ["MVI_40853"], "category": "标准描述"},
    {"query": "卡车压道路实线", "expected": ["MVI_40852"], "category": "标准描述"},
    {"query": "一辆白色轿车", "expected": ["MVI_40855"], "category": "标准描述"},
    {"query": "红灯亮时车辆通过路口", "expected": ["MVI_40853"], "category": "标准描述"},
    {"query": "车辆驶过车道边界", "expected": ["MVI_40852"], "category": "标准描述"},
    {"query": "道路上的白色车辆", "expected": ["MVI_40855"], "category": "标准描述"},
    {"query": "车辆驶入对向车道", "expected": ["MVI_40903"], "category": "标准描述"},
    {"query": "机动车违规反向行驶", "expected": ["MVI_40903"], "category": "标准描述"},
    {"query": "车辆未遵守红绿灯规则", "expected": ["MVI_40853"], "category": "标准描述"},
    {"query": "货车跨越道路标线", "expected": ["MVI_40852"], "category": "标准描述"},
    {"query": "白色机动车", "expected": ["MVI_40855"], "category": "标准描述"},
    {"query": "车辆跨线行驶", "expected": ["MVI_40852"], "category": "标准描述"},
    {"query": "红色车辆违反道路行驶方向", "expected": ["MVI_40854"], "category": "标准描述"},
    # ==================== 2. 同义表达 (20) ====================
    {"query": "红车反方向行驶", "expected": ["MVI_40854"], "category": "同义表达"},
    {"query": "一辆红色车辆朝错误方向开", "expected": ["MVI_40854"], "category": "同义表达"},
    {"query": "红色汽车走错方向了", "expected": ["MVI_40854"], "category": "同义表达"},
    {"query": "红色小汽车逆向通过道路", "expected": ["MVI_40854"], "category": "同义表达"},
    {"query": "汽车反方向行驶", "expected": ["MVI_40903"], "category": "同义表达"},
    {"query": "车辆朝相反方向移动", "expected": ["MVI_40903"], "category": "同义表达"},
    {"query": "车辆错误方向行驶", "expected": ["MVI_40903"], "category": "同义表达"},
    {"query": "汽车逆着车流行驶", "expected": ["MVI_40903"], "category": "同义表达"},
    {"query": "车辆进入反向道路", "expected": ["MVI_40903"], "category": "同义表达"},
    {"query": "车辆无视红灯继续行驶", "expected": ["MVI_40853"], "category": "同义表达"},
    {"query": "机动车闯信号灯", "expected": ["MVI_40853"], "category": "同义表达"},
    {"query": "小车冲过红灯", "expected": ["MVI_40853"], "category": "同义表达"},
    {"query": "车辆抢红灯通过路口", "expected": ["MVI_40853"], "category": "同义表达"},
    {"query": "大型车辆越过车道线", "expected": ["MVI_40852"], "category": "同义表达"},
    {"query": "车辆没有保持车道", "expected": ["MVI_40852"], "category": "同义表达"},
    {"query": "货车偏离正常车道", "expected": ["MVI_40852"], "category": "同义表达"},
    {"query": "机动车压道路标线", "expected": ["MVI_40852"], "category": "同义表达"},
    {"query": "浅色汽车", "expected": ["MVI_40855"], "category": "同义表达"},
    {"query": "一辆颜色较浅的车", "expected": ["MVI_40855"], "category": "同义表达"},
    {"query": "白色车辆经过道路", "expected": ["MVI_40855"], "category": "同义表达"},
    # ==================== 3. 口语表达 (20) ====================
    {"query": "帮我找一辆开反方向的车", "expected": WRONG_WAY, "category": "口语表达"},
    {"query": "看看有没有车闯红灯", "expected": RED_LIGHT, "category": "口语表达"},
    {"query": "有没有车辆不按规定路线走", "expected": WRONG_WAY, "category": "口语表达"},
    {"query": "有没有车压线了", "expected": CROSS_LINE, "category": "口语表达"},
    {"query": "看看哪辆车开得不正常", "expected": EVENT_VIDEOS, "category": "口语表达"},
    {"query": "找红色车逆行的视频", "expected": ["MVI_40854"], "category": "口语表达"},
    {"query": "有没有车在红灯的时候开过去", "expected": RED_LIGHT, "category": "口语表达"},
    {"query": "那个白车在哪", "expected": WHITE_CAR, "category": "口语表达"},
    {"query": "帮我找一下那辆红色的违章车", "expected": ["MVI_40854"], "category": "口语表达"},
    {"query": "大货车压到线了没有", "expected": CROSS_LINE, "category": "口语表达"},
    {"query": "路上有没有车开反了", "expected": WRONG_WAY, "category": "口语表达"},
    {"query": "找一下那个不守红绿灯的", "expected": RED_LIGHT, "category": "口语表达"},
    {"query": "白色那辆车", "expected": WHITE_CAR, "category": "口语表达"},
    {"query": "有没有逆向行驶的车", "expected": WRONG_WAY, "category": "口语表达"},
    {"query": "哪辆车违章了", "expected": EVENT_VIDEOS, "category": "口语表达"},
    {"query": "大车压线了吗", "expected": CROSS_LINE, "category": "口语表达"},
    {"query": "找那个反方向开的车", "expected": WRONG_WAY, "category": "口语表达"},
    {"query": "看看白色轿车的画面", "expected": WHITE_CAR, "category": "口语表达"},
    {"query": "车闯信号灯了吗", "expected": RED_LIGHT, "category": "口语表达"},
    {"query": "有没有车不按规矩开", "expected": EVENT_VIDEOS, "category": "口语表达"},
    # ==================== 4. 复杂组合 (20) ====================
    {"query": "红色汽车发生逆行", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "红色车辆违规行驶", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "白色汽车正常经过道路", "expected": WHITE_CAR, "category": "复杂组合"},
    {"query": "货车在道路上压线", "expected": CROSS_LINE, "category": "复杂组合"},
    {"query": "车辆在路口闯红灯", "expected": RED_LIGHT, "category": "复杂组合"},
    {"query": "红色小汽车反向驶入道路", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "白色轿车在道路上行驶", "expected": WHITE_CAR, "category": "复杂组合"},
    {"query": "卡车越过道路标线行驶", "expected": CROSS_LINE, "category": "复杂组合"},
    {"query": "机动车在红灯时通过路口", "expected": RED_LIGHT, "category": "复杂组合"},
    {"query": "红色车辆在道路上逆向行驶", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "货车没有保持在车道内", "expected": CROSS_LINE, "category": "复杂组合"},
    {"query": "白色小汽车出现在道路上", "expected": WHITE_CAR, "category": "复杂组合"},
    {"query": "车辆在红灯亮起时通过", "expected": RED_LIGHT, "category": "复杂组合"},
    {"query": "机动车反向驶入对向车道", "expected": ["MVI_40903"], "category": "复杂组合"},
    {"query": "找一辆红色的车在逆行", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "货车压着道路标线开", "expected": CROSS_LINE, "category": "复杂组合"},
    {"query": "车辆不按交通信号通行", "expected": RED_LIGHT, "category": "复杂组合"},
    {"query": "机动车在道路上反向行驶", "expected": ["MVI_40903"], "category": "复杂组合"},
    {"query": "红色汽车朝着错误方向行驶", "expected": ["MVI_40854"], "category": "复杂组合"},
    {"query": "汽车在红灯期间通过交叉路口", "expected": RED_LIGHT, "category": "复杂组合"},
    # ==================== 5. 模糊语义 (20) ====================
    {"query": "有车辆违反交通规则", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "道路上存在异常驾驶行为", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "寻找违规车辆", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "车辆行驶状态不正常", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "道路上发生了交通违法", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "有车不按规定路线行驶", "expected": WRONG_WAY, "category": "模糊语义"},
    {"query": "交通违章场景", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "车辆不遵守交通规则", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "路面上有违规行为", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "驾驶行为存在安全隐患", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "不规范驾驶的画面", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "有交通安全问题的片段", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "道路交通异常", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "车辆行为不符合规定", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "交通违法事件", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "违章驾驶视频", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "车辆未按交规行驶", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "道路上有交通违规", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "找有违章的画面", "expected": EVENT_VIDEOS, "category": "模糊语义"},
    {"query": "交通不合规场景", "expected": EVENT_VIDEOS, "category": "模糊语义"},
]

# ---------------------------------------------------------------------------
# 事件元数据加载
# ---------------------------------------------------------------------------

def load_event_metadata() -> Dict[str, List[Dict]]:
    """加载所有视频的事件元数据，返回 {video_stem: [event_dicts]}。"""
    event_dir = PROJECT_ROOT / "metadata" / "events"
    video_events: Dict[str, List[Dict]] = {}

    if not event_dir.exists():
        logger.warning("Event metadata directory not found: %s", event_dir)
        return video_events

    for json_file in sorted(event_dir.glob("*.json")):
        if json_file.name == ".gitkeep":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            video_name = data.get("video_name", "")
            stem = Path(video_name).stem if video_name else json_file.stem
            events = data.get("events", [])
            video_events[stem] = events
        except Exception:
            logger.warning("Failed to load event metadata: %s", json_file)

    logger.info("Loaded event metadata for %s videos", len(video_events))
    return video_events


# ---------------------------------------------------------------------------
# 帧加载与 CN-CLIP 编码（复用 V1 逻辑）
# ---------------------------------------------------------------------------

def load_frame_metadata() -> List[Dict]:
    metadata_path = PROJECT_ROOT / "index" / "frame_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("Loaded %s frame metadata entries", len(metadata))
    return metadata


def resolve_existing_frames(metadata: List[Dict]) -> tuple[List[Path], List[Dict]]:
    frame_paths: List[Path] = []
    valid_metadata: List[Dict] = []
    missing_count = 0

    for item in metadata:
        raw_path = item.get("frame_path", "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            frame_name = Path(raw_path).name
            alt_path = PROJECT_ROOT / "frames" / frame_name
            if alt_path.exists():
                path = alt_path
            else:
                missing_count += 1
                continue
        frame_paths.append(path)
        valid_metadata.append(item)

    if missing_count > 0:
        logger.warning("Skipped %s frames with missing files", missing_count)
    logger.info("Resolved %s existing frame files", len(frame_paths))
    return frame_paths, valid_metadata


def encode_frames_with_cnclip(
    embedding_service: EmbeddingService,
    frame_paths: List[Path],
    batch_size: int = 16,
) -> np.ndarray:
    total = len(frame_paths)
    all_vectors: List[np.ndarray] = []
    logger.info("Encoding %s frames with CN-CLIP (batch_size=%s)...", total, batch_size)
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = frame_paths[batch_start:batch_end]
        records = embedding_service.batch_encode(frame_paths=batch_paths, batch_size=len(batch_paths))
        batch_vectors = np.asarray([r["embedding"] for r in records], dtype=np.float32)
        all_vectors.append(batch_vectors)
        elapsed = time.time() - start_time
        if batch_end % 64 == 0 or batch_end == total:
            logger.info("Encoded %s/%s frames (%.1f%%) elapsed=%.1fs", batch_end, total, batch_end / total * 100, elapsed)

    matrix = np.vstack(all_vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms
    logger.info("Finished encoding %s frames in %.1fs (shape=%s)", total, time.time() - start_time, matrix.shape)
    return matrix


def build_faiss_index(embeddings: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info("Built FAISS index with %s vectors (dim=%s)", index.ntotal, embeddings.shape[1])
    return index


# ---------------------------------------------------------------------------
# 评测核心
# ---------------------------------------------------------------------------

def normalize_video_stem(video_name: str) -> str:
    return Path(str(video_name)).stem.casefold()


def clip_search(
    embedding_service: EmbeddingService,
    index,
    frame_metadata: List[Dict],
    queries: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """用 CN-CLIP 编码多个查询变体，合并 top-K 结果。"""
    import numpy as np

    merged: Dict[str, Dict] = {}
    candidate_k = max(top_k * 3, top_k)

    for q in queries:
        q_vector = embedding_service.encode_text(q)
        q_vector = q_vector.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q_vector, candidate_k)

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(frame_metadata):
                continue
            meta = frame_metadata[idx]
            key = str(meta.get("frame_id") or meta.get("frame_path", ""))
            existing = merged.get(key)
            if existing is None or score > existing["score"]:
                merged[key] = {
                    "rank": 0,  # will re-rank later
                    "score": float(score),
                    "video_name": meta.get("video_name", ""),
                    "timestamp": float(meta.get("timestamp_seconds", meta.get("timestamp", 0.0))),
                    "frame_path": meta.get("frame_path", ""),
                    "matched_query": q,
                }

    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


def dedup_results_by_video(results: List[Dict], top_k: int = 5) -> List[Dict]:
    """视频级去重：每个视频只保留得分最高的一条结果。"""
    best_per_video: Dict[str, Dict] = {}
    for r in results:
        stem = normalize_video_stem(r["video_name"])
        existing = best_per_video.get(stem)
        if existing is None or r["score"] > existing["score"]:
            best_per_video[stem] = r
    deduped = sorted(best_per_video.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for rank, r in enumerate(deduped, start=1):
        r["rank"] = rank
    return deduped


def event_search(
    event_metadata: Dict[str, List[Dict]],
    event_types: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """搜索事件元数据，按事件类型匹配，按置信度排序。

    视频级去重：每个视频只返回置信度最高的一条事件。
    当 event_types 为空时，返回所有事件类型（用于模糊查询的全事件检索）。
    """
    event_type_set = set(event_types)
    # {video_stem: best_event_dict}
    best_per_video: Dict[str, Dict] = {}

    for video_stem, events in event_metadata.items():
        for event in events:
            et = event.get("event_type", "")
            # If event_type_set is non-empty, filter by event types.
            # If empty (vague_event route), search all event types.
            if event_type_set and et not in event_type_set:
                continue
            conf = float(event.get("confidence", 0.0))
            existing = best_per_video.get(video_stem)
            if existing is None or conf > existing["score"]:
                best_per_video[video_stem] = {
                    "score": conf,
                    "video_name": video_stem + ".avi",
                    "timestamp": float(event.get("start_ts", 0.0)),
                    "event_type": et,
                    "event_confidence": conf,
                }

    results = sorted(best_per_video.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def evaluate_query(results: List[Dict], expected_videos: List[str]) -> Dict:
    expected_set: Set[str] = {normalize_video_stem(v) for v in expected_videos}

    hit_at_1 = False
    hit_at_5 = False
    first_relevant_rank = 0
    top1_score = 0.0

    for result in results:
        result_video = normalize_video_stem(result["video_name"])
        rank = result["rank"]
        if rank == 1:
            top1_score = result["score"]
        if result_video in expected_set:
            if rank == 1:
                hit_at_1 = True
            if rank <= 5:
                hit_at_5 = True
            if first_relevant_rank == 0:
                first_relevant_rank = rank

    return {
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "first_relevant_rank": first_relevant_rank,
        "top1_score": top1_score,
    }


def run_mode_a_baseline(
    embedding_service: EmbeddingService,
    index,
    frame_metadata: List[Dict],
    top_k: int = 5,
) -> Dict:
    """Mode A: 纯 CN-CLIP 文本编码 → FAISS 检索（基线）。"""
    query_reports = []
    category_stats: Dict[str, Dict] = {}

    for i, item in enumerate(TEST_QUERIES, start=1):
        query = item["query"]
        expected = item["expected"]
        category = item["category"]

        results = clip_search(embedding_service, index, frame_metadata, [query], top_k * 3)
        results = dedup_results_by_video(results, top_k)
        eval_result = evaluate_query(results, expected)

        report = {
            "query": query, "category": category, "expected_videos": expected,
            "hit_at_1": eval_result["hit_at_1"], "hit_at_5": eval_result["hit_at_5"],
            "first_relevant_rank": eval_result["first_relevant_rank"],
            "top1_score": round(eval_result["top1_score"], 4),
            "top5_results": [{"rank": r["rank"], "video": r["video_name"], "score": round(r["score"], 4)} for r in results],
        }
        query_reports.append(report)

        if category not in category_stats:
            category_stats[category] = {"count": 0, "hit1": 0, "hit5": 0, "rr_sum": 0.0, "score_sum": 0.0}
        cs = category_stats[category]
        cs["count"] += 1
        if eval_result["hit_at_1"]: cs["hit1"] += 1
        if eval_result["hit_at_5"]: cs["hit5"] += 1
        if eval_result["first_relevant_rank"] > 0: cs["rr_sum"] += 1.0 / eval_result["first_relevant_rank"]
        cs["score_sum"] += eval_result["top1_score"]

        status = "OK" if eval_result["hit_at_1"] else ("TOP5" if eval_result["hit_at_5"] else "MISS")
        logger.info("[A][%3d/100] [%s] %s -> %s | top1=%s", i, status, query,
                    results[0]["video_name"] if results else "N/A", eval_result["hit_at_1"])

    return _build_summary("baseline_cnclip", query_reports, category_stats, top_k)


def run_mode_b_tqum(
    embedding_service: EmbeddingService,
    index,
    frame_metadata: List[Dict],
    event_metadata: Dict[str, List[Dict]],
    top_k: int = 5,
) -> Dict:
    """Mode B: TQUM Phase 3 — 置信度分级路由 + Top-k 意图预测。

    Phase 3 升级：使用 route_by_confidence 实现三档置信度路由，
    替代 Phase 2 的二元 skip_non_event 逻辑。

    路由策略：
      - event_primary (conf>=0.8): CLIP + event boost(0.12)
      - hybrid_balanced (conf 0.6-0.8): 加权平均融合
      - clip_primary (conf 0-0.6): CLIP + event boost(0.04)
      - vague_event (无事件+模糊词): CLIP + 全事件 boost(0.06)
      - clip_only (其他): CLIP 主导
    """
    tqum = QueryRewriteService(use_chinese_clip=True)

    # 轻量包装器用于 route_by_confidence
    class RouteHelper:
        route_by_confidence = HybridSearchService.route_by_confidence
        _intent_value = HybridSearchService._intent_value

    route_helper = RouteHelper()

    query_reports = []
    category_stats: Dict[str, Dict] = {}

    for i, item in enumerate(TEST_QUERIES, start=1):
        query = item["query"]
        expected = item["expected"]
        category = item["category"]

        # TQUM 解析
        intent = tqum.parse_query_intent(query)
        query_type = intent.query_type
        event_types = intent.event_types
        event_conf = intent.event_confidence
        rewrites = intent.rewritten_queries
        attributes = intent.attributes if hasattr(intent, "attributes") else {}
        intent_candidates = intent.intent_candidates if hasattr(intent, "intent_candidates") else []

        # Phase 3: 置信度分级路由
        route_info = route_helper.route_by_confidence(intent)
        route = route_info["route"]

        has_visual_attr = bool(attributes.get("color") or attributes.get("vehicle_type"))

        results: List[Dict] = []
        channel = route

        VISUAL_TOKENS = {"红色", "白色", "黑色", "蓝色", "黄色", "浅色",
                         "汽车", "轿车", "货车", "卡车", "公交车", "巴士",
                         "摩托车", "机动", "大车", "小车", "红灯", "压线", "逆行"}

        if route in ("event_primary", "clip_primary", "vague_event"):
            # CLIP 主排序 + 事件 boost 策略
            event_boost = route_info.get("event_boost", 0.02)

            # 事件搜索：vague_event 时搜索全部事件类型
            search_types = event_types if route != "vague_event" else []
            event_candidates = event_search(event_metadata, search_types, top_k=top_k * 4) if (search_types or route == "vague_event") else []

            # 建立 video_stem -> event_confidence 映射
            event_confs: Dict[str, float] = {}
            for ec in event_candidates:
                stem = normalize_video_stem(ec["video_name"])
                event_confs[stem] = max(event_confs.get(stem, 0.0), ec["event_confidence"])

            # CLIP 查询筛选
            if has_visual_attr:
                clip_queries = [r for r in rewrites if any(tok in r for tok in VISUAL_TOKENS)]
                if not clip_queries:
                    clip_queries = rewrites[:4]
            else:
                clip_queries = [query] + [r for r in rewrites if r != query][:3]

            clip_results = clip_search(
                embedding_service, index, frame_metadata, clip_queries, top_k * 4
            )
            clip_results = dedup_results_by_video(clip_results, top_k * 4)

            # CLIP 主排序 + 事件动态 boost
            reranked = []
            for cr in clip_results:
                stem = normalize_video_stem(cr["video_name"])
                ec = event_confs.get(stem, 0.0)
                combined = cr["score"] + event_boost * ec
                reranked.append({
                    "score": combined,
                    "video_name": cr["video_name"],
                    "clip_score": cr["score"],
                    "event_confidence": ec,
                })

            # 补充事件-only 结果（不在 CLIP 结果中的事件视频）
            clip_stems = {normalize_video_stem(cr["video_name"]) for cr in clip_results}
            for ec in event_candidates:
                stem = normalize_video_stem(ec["video_name"])
                if stem not in clip_stems:
                    combined = event_boost * ec["event_confidence"]
                    reranked.append({
                        "score": combined,
                        "video_name": ec["video_name"],
                        "clip_score": 0.0,
                        "event_confidence": ec["event_confidence"],
                    })

            reranked.sort(key=lambda x: x["score"], reverse=True)
            results = reranked[:top_k]

        else:
            # clip_only: CLIP 中文扩展
            results = clip_search(
                embedding_service, index, frame_metadata, rewrites, top_k * 3
            )
            results = dedup_results_by_video(results, top_k)

        # Assign ranks
        for rank_idx, r in enumerate(results, start=1):
            r["rank"] = rank_idx

        eval_result = evaluate_query(results, expected)

        report = {
            "query": query, "category": category, "expected_videos": expected,
            "hit_at_1": eval_result["hit_at_1"], "hit_at_5": eval_result["hit_at_5"],
            "first_relevant_rank": eval_result["first_relevant_rank"],
            "top1_score": round(eval_result["top1_score"], 4),
            "tqum_type": query_type, "tqum_events": event_types,
            "tqum_conf": event_conf, "channel": channel,
            "route": route,
            "has_visual_attr": has_visual_attr,
            "intent_candidates": intent_candidates,
            "rewrites": rewrites[:5],
            "top5_results": [{"rank": r["rank"], "video": r["video_name"], "score": round(r["score"], 4)} for r in results],
        }
        query_reports.append(report)

        if category not in category_stats:
            category_stats[category] = {"count": 0, "hit1": 0, "hit5": 0, "rr_sum": 0.0, "score_sum": 0.0}
        cs = category_stats[category]
        cs["count"] += 1
        if eval_result["hit_at_1"]: cs["hit1"] += 1
        if eval_result["hit_at_5"]: cs["hit5"] += 1
        if eval_result["first_relevant_rank"] > 0: cs["rr_sum"] += 1.0 / eval_result["first_relevant_rank"]
        cs["score_sum"] += eval_result["top1_score"]

        status = "OK" if eval_result["hit_at_1"] else ("TOP5" if eval_result["hit_at_5"] else "MISS")
        logger.info("[B][%3d/100] [%s] %s -> %s | route=%s conf=%.1f",
                    i, status, query,
                    results[0]["video_name"] if results else "N/A",
                    route, event_conf)

    return _build_summary("tqum_phase3", query_reports, category_stats, top_k)


def _build_summary(mode_name: str, query_reports: List[Dict], category_stats: Dict, top_k: int) -> Dict:
    total_count = len(query_reports)
    total_hit1 = sum(1 for r in query_reports if r["hit_at_1"])
    total_hit5 = sum(1 for r in query_reports if r["hit_at_5"])
    total_rr = sum((1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] > 0 else 0.0 for r in query_reports)
    total_score = sum(r["top1_score"] for r in query_reports)

    per_category = []
    for cat, cs in category_stats.items():
        per_category.append({
            "category": cat, "count": cs["count"],
            "recall_at_1": round(cs["hit1"] / cs["count"], 4),
            "recall_at_5": round(cs["hit5"] / cs["count"], 4),
            "mrr": round(cs["rr_sum"] / cs["count"], 4),
            "avg_score": round(cs["score_sum"] / cs["count"], 4),
        })

    return {
        "mode": mode_name, "total_queries": total_count, "top_k": top_k,
        "overall": {
            "recall_at_1": round(total_hit1 / total_count, 4),
            "recall_at_5": round(total_hit5 / total_count, 4),
            "mrr": round(total_rr / total_count, 4),
            "avg_score": round(total_score / total_count, 4),
        },
        "per_category": per_category,
        "query_reports": query_reports,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    settings.clip_backend = "cnclip"
    settings.cnclip_model_name = "ViT-B-16"

    logger.info("TQUM comparative eval starting (model=%s)", settings.cnclip_model_name)

    # 1. 加载帧元数据
    all_metadata = load_frame_metadata()
    frame_paths, valid_metadata = resolve_existing_frames(all_metadata)

    # 2. 加载事件元数据
    event_metadata = load_event_metadata()

    # 3. CN-CLIP 编码所有帧（编码一次，两种模式共用）
    embedding_service = EmbeddingService(settings)
    embeddings = encode_frames_with_cnclip(embedding_service, frame_paths, batch_size=16)

    # 4. 构建 FAISS 内存索引
    index = build_faiss_index(embeddings)

    # 5. Mode A: 基线（纯 CN-CLIP）
    logger.info("=" * 60)
    logger.info("Mode A: Baseline (pure CN-CLIP)")
    logger.info("=" * 60)
    report_a = run_mode_a_baseline(embedding_service, index, valid_metadata, top_k=5)

    # 6. Mode B: TQUM
    logger.info("=" * 60)
    logger.info("Mode B: TQUM + CN-CLIP")
    logger.info("=" * 60)
    report_b = run_mode_b_tqum(embedding_service, index, valid_metadata, event_metadata, top_k=5)

    # 7. 输出对比
    print("\n" + "=" * 70)
    print("CN-CLIP vs TQUM 对比评测结果")
    print("=" * 70)

    for label, report in [("Mode A (Baseline CN-CLIP)", report_a), ("Mode B (TQUM + CN-CLIP)", report_b)]:
        o = report["overall"]
        print(f"\n【{label}】")
        print(f"  Recall@1       = {o['recall_at_1']:.2%} ({int(o['recall_at_1'] * 100)}/100)")
        print(f"  Recall@5       = {o['recall_at_5']:.2%} ({int(o['recall_at_5'] * 100)}/100)")
        print(f"  MRR            = {o['mrr']:.4f}")
        print(f"  Average Score  = {o['avg_score']:.4f}")
        print(f"  {'类别':<10} {'R@1':>8} {'R@5':>8} {'MRR':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for cat in report["per_category"]:
            print(f"  {cat['category']:<10} {cat['recall_at_1']:>7.0%} {cat['recall_at_5']:>7.0%} {cat['mrr']:>8.4f}")

    # 8. 提升幅度
    o_a = report_a["overall"]
    o_b = report_b["overall"]
    print(f"\n【提升幅度】")
    print(f"  Recall@1: {o_a['recall_at_1']:.0%} -> {o_b['recall_at_1']:.0%} (+{(o_b['recall_at_1'] - o_a['recall_at_1'])*100:.0f}pp)")
    print(f"  Recall@5: {o_a['recall_at_5']:.0%} -> {o_b['recall_at_5']:.0%} (+{(o_b['recall_at_5'] - o_a['recall_at_5'])*100:.0f}pp)")
    print(f"  MRR:      {o_a['mrr']:.4f} -> {o_b['mrr']:.4f} (+{o_b['mrr'] - o_a['mrr']:.4f})")

    # 9. 保存报告
    output_path = PROJECT_ROOT.parent / "tqum_comparative_eval_report.json"
    output_path.write_text(
        json.dumps({"mode_a_baseline": report_a, "mode_b_tqum": report_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Report saved to %s", output_path)

    # 10. 输出 TQUM 路由统计
    from collections import Counter
    route_counts = Counter(r.get("route", "unknown") for r in report_b["query_reports"])
    attr_count = sum(1 for r in report_b["query_reports"] if r.get("has_visual_attr"))
    vague_count = sum(1 for r in report_b["query_reports"] if any(c.get("vague") for c in r.get("intent_candidates", [])))
    print(f"\n【TQUM Phase 3 路由统计】")
    for rt, cnt in route_counts.most_common():
        print(f"  {rt}: {cnt}/100 条")
    print(f"  含视觉属性: {attr_count}/100 条")
    print(f"  含模糊意图: {vague_count}/100 条")


if __name__ == "__main__":
    main()
