#!/usr/bin/env python
"""
CN-CLIP 中文语义泛化评测脚本

评测目标：验证 CN-CLIP ViT-B-16 对中文自然语言、多样表达、同义描述、
隐含语义的理解能力，而不是简单关键词匹配。

评测方法：
  1. 用 CN-CLIP ViT-B-16 重新编码所有帧图像（现有索引为 OpenCLIP，需重建）
  2. 构建 FAISS IndexFlatIP 内存索引
  3. 对 100 条中文查询逐一用 CN-CLIP 编码文本，检索 top-5 帧级结果
  4. 检查期望视频是否出现在 top-1 / top-5 结果中
  5. 统计 Recall@1, Recall@5, MRR, Average CLIP Score

注意：本评测直接使用 CN-CLIP 文本编码，不经过 QueryRewriteService 改写，
确保测试的是 CN-CLIP 模型本身的中文语义能力。

测试集分布（100 条）：
  - 标准描述  20 条：直接、清晰的事件描述
  - 同义表达  20 条：同一事件的不同表达方式
  - 口语表达  20 条：日常口语化查询
  - 复杂组合  20 条：事件+属性复合查询
  - 模糊语义  20 条：抽象、不指定具体事件的查询
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from services.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 视频标签映射
# ---------------------------------------------------------------------------
# MVI_40852: 货车压线
# MVI_40853: 车辆闯红灯
# MVI_40854: 红色汽车逆行
# MVI_40855: 白色汽车（正常行驶）
# MVI_40903: 机动车逆向行驶

WRONG_WAY = ["MVI_40854", "MVI_40903"]
RED_LIGHT = ["MVI_40853"]
CROSS_LINE = ["MVI_40852"]
WHITE_CAR = ["MVI_40855"]
EVENT_VIDEOS = ["MVI_40852", "MVI_40853", "MVI_40854", "MVI_40903"]

# ---------------------------------------------------------------------------
# 100 条测试集
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
# 帧加载与 CN-CLIP 编码
# ---------------------------------------------------------------------------

def load_frame_metadata() -> List[Dict]:
    """从 frame_metadata.json 加载所有帧的元数据。"""
    metadata_path = PROJECT_ROOT / "index" / "frame_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("Loaded %s frame metadata entries from %s", len(metadata), metadata_path.name)
    return metadata


def resolve_existing_frames(metadata: List[Dict]) -> tuple[List[Path], List[Dict]]:
    """过滤出磁盘上实际存在的帧文件，返回 (帧路径列表, 对应元数据列表)。"""
    frame_paths: List[Path] = []
    valid_metadata: List[Dict] = []
    missing_count = 0

    for item in metadata:
        raw_path = item.get("frame_path", "")
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            # 尝试在项目 frames 目录下查找（路径大小写或分隔符差异）
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
    """用 CN-CLIP 批量编码所有帧图像，返回归一化后的向量矩阵。"""
    total = len(frame_paths)
    all_vectors: List[np.ndarray] = []

    logger.info("Encoding %s frames with CN-CLIP (batch_size=%s)...", total, batch_size)
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = frame_paths[batch_start:batch_end]

        records = embedding_service.batch_encode(
            frame_paths=batch_paths,
            batch_size=len(batch_paths),
        )
        batch_vectors = np.asarray(
            [record["embedding"] for record in records], dtype=np.float32
        )
        all_vectors.append(batch_vectors)

        elapsed = time.time() - start_time
        progress = batch_end / total * 100
        speed = batch_end / elapsed if elapsed > 0 else 0
        logger.info(
            "Encoded %s/%s frames (%.1f%%) | elapsed=%.1fs speed=%.1f frames/s",
            batch_end, total, progress, elapsed, speed,
        )

    matrix = np.vstack(all_vectors).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    elapsed = time.time() - start_time
    logger.info(
        "Finished encoding %s frames in %.1fs (shape=%s)",
        total, elapsed, matrix.shape,
    )
    return matrix


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """构建 FAISS IndexFlatIP 内存索引。"""
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info(
        "Built FAISS IndexFlatIP with %s vectors (dim=%s)",
        index.ntotal, embeddings.shape[1],
    )
    return index


# ---------------------------------------------------------------------------
# 查询评测
# ---------------------------------------------------------------------------

def normalize_video_stem(video_name: str) -> str:
    """提取视频名的主干（去掉扩展名，统一小写）。"""
    return Path(str(video_name)).stem.casefold()


def search_query(
    embedding_service: EmbeddingService,
    index: "faiss.Index",
    frame_metadata: List[Dict],
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """用 CN-CLIP 编码查询文本，检索 top-K 帧级结果。"""
    query_vector = embedding_service.encode_text(query)
    query_vector = query_vector.reshape(1, -1).astype(np.float32)

    # FAISS 内积检索（向量已归一化，内积 = 余弦相似度）
    scores, indices = index.search(query_vector, top_k)

    results: List[Dict] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(frame_metadata):
            continue
        meta = frame_metadata[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "video_name": meta.get("video_name", ""),
            "timestamp": float(meta.get("timestamp_seconds", meta.get("timestamp", 0.0))),
            "frame_path": meta.get("frame_path", ""),
        })
    return results


def evaluate_query(
    results: List[Dict],
    expected_videos: List[str],
) -> Dict:
    """评估单条查询的检索结果。

    返回:
        hit_at_1: top-1 结果是否命中期望视频
        hit_at_5: top-5 结果中是否有期望视频
        first_relevant_rank: 第一个命中结果的排名（从 1 开始，未命中为 0）
        top1_score: top-1 的 CLIP 相似度分数
    """
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


def run_evaluation(
    embedding_service: EmbeddingService,
    index: "faiss.Index",
    frame_metadata: List[Dict],
    top_k: int = 5,
) -> Dict:
    """运行全部 100 条查询的评测。"""
    query_reports: List[Dict] = []
    category_stats: Dict[str, Dict] = {}

    total = len(TEST_QUERIES)
    logger.info("Starting evaluation of %s queries (top_k=%s)...", total, top_k)

    for i, item in enumerate(TEST_QUERIES, start=1):
        query = item["query"]
        expected = item["expected"]
        category = item["category"]

        results = search_query(
            embedding_service=embedding_service,
            index=index,
            frame_metadata=frame_metadata,
            query=query,
            top_k=top_k,
        )
        eval_result = evaluate_query(results, expected)

        report = {
            "query": query,
            "category": category,
            "expected_videos": expected,
            "hit_at_1": eval_result["hit_at_1"],
            "hit_at_5": eval_result["hit_at_5"],
            "first_relevant_rank": eval_result["first_relevant_rank"],
            "top1_score": round(eval_result["top1_score"], 4),
            "top5_results": [
                {
                    "rank": r["rank"],
                    "video": r["video_name"],
                    "score": round(r["score"], 4),
                    "ts": r["timestamp"],
                }
                for r in results
            ],
        }
        query_reports.append(report)

        # 更新分类统计
        if category not in category_stats:
            category_stats[category] = {
                "count": 0, "hit1": 0, "hit5": 0,
                "rr_sum": 0.0, "score_sum": 0.0,
            }
        cs = category_stats[category]
        cs["count"] += 1
        if eval_result["hit_at_1"]:
            cs["hit1"] += 1
        if eval_result["hit_at_5"]:
            cs["hit5"] += 1
        if eval_result["first_relevant_rank"] > 0:
            cs["rr_sum"] += 1.0 / eval_result["first_relevant_rank"]
        cs["score_sum"] += eval_result["top1_score"]

        status = "OK" if eval_result["hit_at_1"] else ("TOP5" if eval_result["hit_at_5"] else "MISS")
        logger.info(
            "[%3d/%d] [%s] %s → %s | top1=%s score=%.4f",
            i, total, status, query,
            results[0]["video_name"] if results else "N/A",
            eval_result["hit_at_1"],
            eval_result["top1_score"],
        )

    # 汇总指标
    total_count = len(query_reports)
    total_hit1 = sum(1 for r in query_reports if r["hit_at_1"])
    total_hit5 = sum(1 for r in query_reports if r["hit_at_5"])
    total_rr = sum(
        (1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] > 0 else 0.0
        for r in query_reports
    )
    total_score = sum(r["top1_score"] for r in query_reports)

    recall_at_1 = total_hit1 / total_count if total_count else 0.0
    recall_at_5 = total_hit5 / total_count if total_count else 0.0
    mrr = total_rr / total_count if total_count else 0.0
    avg_score = total_score / total_count if total_count else 0.0

    # 分类汇总
    per_category: List[Dict] = []
    for cat, cs in category_stats.items():
        per_category.append({
            "category": cat,
            "count": cs["count"],
            "recall_at_1": round(cs["hit1"] / cs["count"], 4),
            "recall_at_5": round(cs["hit5"] / cs["count"], 4),
            "mrr": round(cs["rr_sum"] / cs["count"], 4),
            "avg_score": round(cs["score_sum"] / cs["count"], 4),
        })

    return {
        "model": "CN-CLIP ViT-B-16",
        "total_queries": total_count,
        "top_k": top_k,
        "overall": {
            "recall_at_1": round(recall_at_1, 4),
            "recall_at_5": round(recall_at_5, 4),
            "mrr": round(mrr, 4),
            "avg_clip_score": round(avg_score, 4),
        },
        "per_category": per_category,
        "query_reports": query_reports,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()

    # 确认使用 CN-CLIP 后端
    backend = settings.clip_backend.lower()
    if backend != "cnclip":
        logger.warning(
            "clip_backend=%s (expected cnclip). Forcing cnclip for this eval.",
            backend,
        )
        settings.clip_backend = "cnclip"
        settings.cnclip_model_name = "ViT-B-16"

    logger.info("CN-CLIP semantic eval starting (model=%s)", settings.cnclip_model_name)

    # 1. 加载帧元数据
    all_metadata = load_frame_metadata()

    # 2. 过滤出实际存在的帧文件
    frame_paths, valid_metadata = resolve_existing_frames(all_metadata)

    # 3. 加载 CN-CLIP 模型并编码所有帧
    embedding_service = EmbeddingService(settings)
    embeddings = encode_frames_with_cnclip(embedding_service, frame_paths, batch_size=16)

    # 4. 构建 FAISS 内存索引
    index = build_faiss_index(embeddings)

    # 5. 运行 100 条查询评测
    report = run_evaluation(
        embedding_service=embedding_service,
        index=index,
        frame_metadata=valid_metadata,
        top_k=5,
    )

    # 6. 输出汇总
    print("\n" + "=" * 60)
    print("CN-CLIP 中文语义泛化评测结果")
    print("=" * 60)
    print(f"模型: {report['model']}")
    print(f"查询数: {report['total_queries']}")
    print(f"Top-K: {report['top_k']}")
    print()
    print("【总体指标】")
    o = report["overall"]
    print(f"  Recall@1       = {o['recall_at_1']:.2%} ({int(o['recall_at_1'] * report['total_queries'])}/{report['total_queries']})")
    print(f"  Recall@5       = {o['recall_at_5']:.2%} ({int(o['recall_at_5'] * report['total_queries'])}/{report['total_queries']})")
    print(f"  MRR            = {o['mrr']:.4f}")
    print(f"  Average Score  = {o['avg_clip_score']:.4f}")
    print()
    print("【分类指标】")
    print(f"  {'类别':<10} {'数量':>4} {'R@1':>8} {'R@5':>8} {'MRR':>8} {'AvgScore':>10}")
    print(f"  {'-'*10} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for cat in report["per_category"]:
        print(
            f"  {cat['category']:<10} {cat['count']:>4} "
            f"{cat['recall_at_1']:>7.1%} {cat['recall_at_5']:>7.1%} "
            f"{cat['mrr']:>8.4f} {cat['avg_score']:>10.4f}"
        )
    print()

    # 7. 保存详细报告
    output_path = PROJECT_ROOT.parent / "cnclip_semantic_eval_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Detailed report saved to %s", output_path)

    # 8. 输出未命中查询
    misses = [r for r in report["query_reports"] if not r["hit_at_5"]]
    if misses:
        print(f"\n【未命中查询 (top-5 全部 miss): {len(misses)} 条】")
        for r in misses:
            top1 = r["top5_results"][0] if r["top5_results"] else {}
            print(f"  [{r['category']}] '{r['query']}'")
            print(f"    期望: {r['expected_videos']} | top1: {top1.get('video','N/A')} (score={top1.get('score',0)})")


if __name__ == "__main__":
    main()
