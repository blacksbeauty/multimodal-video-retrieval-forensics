from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence

from config import Settings


logger = logging.getLogger(__name__)


class ResultAggregationService:
    """检索结果聚合服务：将多通道帧级结果聚合成去重的段级结果。

    聚合规则（S7）：
      - 段级结果（含 segment_id）本身已是聚合单元 → 直接按 segment_id 去重；
      - 帧级结果 → 按 (video_id, 5s 时间窗) 分组，每组保留最高分记录；
      - 低于 hybrid_score_threshold 的结果直接丢弃；
      - 输出按 (-best_score, video_name, start_ts) 排序并截断到 top_k。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def aggregate_results(
        self,
        results: Sequence[Dict],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[Dict]:
        """Group results by video and time window, keeping the best-scoring frame per segment.

        S7: 段级结果（含 segment_id）本身已是聚合单元，直接按 segment_id 去重，
        不再按时间窗二次聚合（避免两个段中点落同一 5s 桶互相覆盖）。
        """
        threshold = self.settings.hybrid_score_threshold if score_threshold is None else score_threshold
        grouped: Dict[tuple[str, int], Dict] = {}
        grouped_by_segment: Dict[str, Dict] = {}

        for item in results:
            score = float(item.get("score", item.get("best_score", 0.0)))
            if score < threshold:
                continue

            segment_id = str(item.get("segment_id", ""))
            if segment_id:
                normalized = self._normalize_result(item, score)
                existing_seg = grouped_by_segment.get(segment_id)
                if existing_seg is None or normalized["best_score"] > existing_seg["best_score"]:
                    grouped_by_segment[segment_id] = normalized
                continue

            normalized = self._normalize_result(item, score)
            window_index = int(math.floor(normalized["timestamp_seconds"] / self.settings.segment_window_seconds))
            group_key = (normalized["video_id"], window_index)

            existing = grouped.get(group_key)
            if existing is None or normalized["best_score"] > existing["best_score"]:
                grouped[group_key] = normalized
            elif normalized["best_score"] == existing["best_score"]:
                existing["matched_by"] = sorted(set(existing["matched_by"]) | set(normalized["matched_by"]))

        merged: Dict = {}
        for key, value in grouped.items():
            merged.setdefault(value["video_id"] + ":" + str(key[1]), value)
        for segment_id, value in grouped_by_segment.items():
            merged[f"seg:{segment_id}"] = value

        aggregated = sorted(
            merged.values(),
            key=lambda item: (-item["best_score"], item["video_name"], item["start_ts"]),
        )
        logger.info(
            "Aggregated retrieval results input=%s output=%s threshold=%s",
            len(results),
            len(aggregated),
            threshold,
        )
        return aggregated[:top_k]

    def _normalize_result(self, item: Dict, score: float) -> Dict:
        """把一条通道结果规范化为统一的段级记录（字段全量补齐，缺省为空/0）。

        关键兜底：
          - end_ts <= start_ts 时强制补 1 秒跨度（防止产出 0 秒不可播放片段）；
          - matched_by 兼容字符串与列表两种形态；
          - thumbnail_frame 缺省回退 frame_path，保证前端有图可显示。
        """
        timestamp_seconds = float(item.get("timestamp_seconds", item.get("timestamp", item.get("start_ts", 0.0))))
        window_start = (
            math.floor(timestamp_seconds / self.settings.segment_window_seconds)
            * self.settings.segment_window_seconds
        )
        window_end = window_start + self.settings.segment_window_seconds

        matched_by = item.get("matched_by", ["clip"])
        if isinstance(matched_by, str):
            matched_by = [matched_by]

        start_ts = float(item.get("start_ts", window_start))
        end_ts = float(item.get("end_ts", max(window_end, timestamp_seconds)))
        if end_ts <= start_ts:
            # 单点/无效时间段（如轨迹单帧、起止时间相同的通道结果）：
            # 至少补 1 秒跨度，避免产出 0 秒剪辑（播放器无法播放）。
            end_ts = start_ts + 1.0

        return {
            "video_id": str(item.get("video_id", "")),
            "video_name": str(item.get("video_name", "")),
            "video_path": str(item.get("video_path", "")),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "best_score": score,
            "matched_by": sorted({str(value) for value in matched_by}),
            "clip_score": float(item.get("clip_score", 0.0)),
            "detection_score": float(item.get("detection_score", 0.0)),
            "trajectory_score": float(item.get("trajectory_score", 0.0)),
            "event_score": float(item.get("event_score", 0.0)),
            "matched_label": str(item.get("matched_label", "")),
            "matched_direction": str(item.get("matched_direction", "")),
            "matched_event_type": str(item.get("matched_event_type", "")),
            "track_id": str(item.get("track_id", "")),
            "event_id": str(item.get("event_id", "")),
            "thumbnail_frame": str(item.get("thumbnail_frame", item.get("frame_path", ""))),
            "frame_id": str(item.get("frame_id", "")),
            # 取证三帧快照透传（非事件通道结果为空列表）
            "key_snapshots": list(item.get("key_snapshots") or []),
            # S7: 段级字段透传（帧级结果为空字符串）
            "segment_id": str(item.get("segment_id", "")),
            "timestamp_seconds": timestamp_seconds,
        }
