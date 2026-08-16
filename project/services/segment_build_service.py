from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

from config import Settings
from services.segment_service import SegmentService


logger = logging.getLogger(__name__)


class SegmentBuildService:
    """S7: 段级索引生成编排（数据规范 v1.1 §3.8 + 段级FAISS索引适配方案）。

    职责:
      1. build_video_segments(): 事件驱动段（时间窗重叠事件合并）+ 5s 兜底段
      2. ingest_segment_pipeline(): 段文本编码 → 段级 FAISS upsert → segments 落盘
      3. 更新 events 元数据 segment_built 标记（异步触发，失败不影响帧级索引）
    """

    def __init__(self, settings: Settings, segment_service: SegmentService | None = None) -> None:
        self.settings = settings
        self.segment_service = segment_service or SegmentService(settings)

    # ------------------------------------------------------------------ #
    # 段构建
    # ------------------------------------------------------------------ #

    def build_video_segments(
        self,
        video_id: str,
        video_name: str,
        video_path: str,
        events: Sequence[Dict],
        duration_sec: float | None = None,
        frame_paths_by_time: Dict[float, str] | None = None,
    ) -> List[Dict]:
        """构建一个视频的段列表（事件驱动段 + 5s 兜底段，事件区间不重叠）。

        合并规则（评审 #2 定稿）:
          - 按 start_ts 升序处理；
          - 事件"相邻或重叠"（event_j.start_ts <= event_i.end_ts + EPSILON）合并为一段，
            description 用 "；" 拼接，时间窗取 [min(start), max(end)] 不外扩；
          - 未被事件覆盖的 [0, duration] 区间用 5s 窗口切兜底段。
        """
        if not events and duration_sec is None:
            raise ValueError("events 与 duration_sec 至少提供一个。")

        sorted_events = sorted(
            events, key=lambda item: (float(item["start_ts"]), float(item["end_ts"]))
        )

        # ---- 1. 事件驱动段：合并重叠/相邻事件 ----
        # 判定语义：事件与上一组合并组的时间窗"相邻或重叠"（start <= last_end + eps）时
        # 应并入上一组；仅当明显分离（start > last_end + eps）才另起新组。
        # （Code Review Must Fix #1：已修复，原条件方向相反会把相邻事件拆段）
        merged_groups: List[List[Dict]] = []
        for event in sorted_events:
            start = float(event["start_ts"])
            if merged_groups:
                last_group = merged_groups[-1]
                last_end = max(float(e["end_ts"]) for e in last_group)
                if start > last_end + self.settings.timestamp_epsilon:
                    merged_groups.append([event])
                else:
                    last_group.append(event)
            else:
                merged_groups.append([event])

        segments: List[Dict] = []
        for index, group in enumerate(merged_groups, start=1):
            start_ts = min(float(e["start_ts"]) for e in group)
            end_ts = max(float(e["end_ts"]) for e in group)
            descriptions = [e.get("description", "") for e in group if e.get("description")]
            text = self.segment_service.build_segment_text(
                video_id=video_id,
                start_ts=start_ts,
                description="；".join(descriptions) if descriptions else "",
            )
            evidence_frames = [
                path for e in group for path in (e.get("evidence_frames") or [])
            ]
            segments.append(
                {
                    "segment_id": f"{video_id}_seg{index:03d}",
                    "video_id": video_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "time_range": {"start": round(start_ts, 2), "end": round(end_ts, 2)},
                    "frame_paths": list(dict.fromkeys(evidence_frames)),
                    "events": [e["event_type"] for e in group],
                    "event_ids": [e["event_id"] for e in group],
                    "tracks": sorted({t for e in group for t in (e.get("track_ids") or [])}),
                    "text": text,
                    "text_source": "事件描述模板",
                }
            )

        # ---- 2. 兜底段：未被事件覆盖的区间按 5s 切分 ----
        duration = duration_sec if duration_sec is not None else max(
            (float(e["end_ts"]) for e in sorted_events), default=0.0
        )
        covered = [(float(e["start_ts"]), float(e["end_ts"])) for e in sorted_events]
        segments.extend(
            self._build_fallback_segments(
                video_id=video_id,
                video_name=video_name,
                video_path=video_path,
                duration=duration,
                covered=covered,
                start_index=len(segments) + 1,
                frame_paths_by_time=frame_paths_by_time or {},
            )
        )

        segments.sort(key=lambda item: (item["time_range"]["start"], item["segment_id"]))
        logger.info(
            "Built segments video_id=%s event_segments=%s fallback_segments=%s total=%s",
            video_id,
            len(merged_groups),
            len(segments) - len(merged_groups),
            len(segments),
        )
        return segments

    def _build_fallback_segments(
        self,
        video_id: str,
        video_name: str,
        video_path: str,
        duration: float,
        covered: List[tuple[float, float]],
        start_index: int,
        frame_paths_by_time: Dict[float, str],
    ) -> List[Dict]:
        """未被事件覆盖的 [0, duration] 区间按 5s 窗口切兜底段。"""
        window = float(self.settings.segment_window_seconds)
        covered_sorted = sorted(covered)
        segments: List[Dict] = []
        cursor = 0.0

        for start, end in covered_sorted:
            # 事件前的空隙
            while cursor + window <= start + self.settings.timestamp_epsilon:
                seg_end = min(cursor + window, start)
                segments.append(
                    self._fallback_segment(
                        video_id, video_name, video_path, cursor, seg_end,
                        len(segments) + start_index, frame_paths_by_time,
                    )
                )
                cursor = seg_end
            cursor = max(cursor, end)

        # 最后一个事件之后的空隙
        while cursor + window <= duration + self.settings.timestamp_epsilon:
            seg_end = min(cursor + window, duration)
            segments.append(
                self._fallback_segment(
                    video_id, video_name, video_path, cursor, seg_end,
                    len(segments) + start_index, frame_paths_by_time,
                )
            )
            cursor = seg_end

        # 若视频极短（< 窗口）且无事件覆盖，兜底一段
        if not segments and duration > 0:
            segments.append(
                self._fallback_segment(
                    video_id, video_name, video_path, 0.0, duration,
                    start_index, frame_paths_by_time,
                )
            )
        return segments

    def _fallback_segment(
        self,
        video_id: str,
        video_name: str,
        video_path: str,
        start_ts: float,
        end_ts: float,
        index: int,
        frame_paths_by_time: Dict[float, str],
    ) -> Dict:
        start_ts = round(start_ts, 2)
        end_ts = round(end_ts, 2)
        text = self.segment_service.build_segment_text(
            video_id=video_id,
            start_ts=start_ts,
            description="",
        )
        in_window = [
            path for ts, path in sorted(frame_paths_by_time.items())
            if start_ts - self.settings.timestamp_epsilon <= ts <= end_ts + self.settings.timestamp_epsilon
        ]
        return {
            "segment_id": f"{video_id}_seg{index:03d}",
            "video_id": video_id,
            "video_name": video_name,
            "video_path": video_path,
            "time_range": {"start": start_ts, "end": end_ts},
            "frame_paths": in_window[:3],
            "events": [],
            "event_ids": [],
            "tracks": [],
            "text": text,
            "text_source": "时间窗兜底",
        }

    # ------------------------------------------------------------------ #
    # 摄入流水线
    # ------------------------------------------------------------------ #

    def ingest_segment_pipeline(
        self,
        video_id: str,
        clip_service,
        index_service,
    ) -> Dict[str, object]:
        """S7: 段级全流程——构建段 → 文本编码 → FAISS upsert → segments 落盘 → 标记。

        失败不抛出（记录日志并保持 segment_built=false），由调用方决定重试；
        帧级索引与检索不受影响（P5 降级，评审 #4）。
        """
        events_payload = self._load_events(video_id)
        if events_payload is None:
            logger.warning("No event metadata for video_id=%s; skipping segment build", video_id)
            return {"video_id": video_id, "built": False, "reason": "no_events"}

        events = events_payload.get("events", [])
        video_name = events_payload.get("video_name", video_id)
        video_path = events_payload.get("video_path", "")

        try:
            segments = self.build_video_segments(
                video_id=video_id,
                video_name=video_name,
                video_path=video_path,
                events=events,
            )
        except Exception:
            logger.exception("Failed to build segments for video_id=%s", video_id)
            return {"video_id": video_id, "built": False, "reason": "build_failed"}

        if not segments:
            logger.info("No segments generated for video_id=%s", video_id)
            return {"video_id": video_id, "built": False, "reason": "empty_segments"}

        try:
            # 段文本编码（与查询同空间：encode_text）
            vectors = self._encode_segment_texts(clip_service, segments)
            # 段级 FAISS upsert
            index_service.upsert_segment_records(
                video_id=video_id,
                segments=segments,
                embeddings=vectors,
            )
            # segments 元数据 + 段向量 meta 落盘
            embeddings_meta = [
                {
                    "segment_id": seg["segment_id"],
                    "model": "CN-CLIP" if getattr(self.settings, "clip_backend", "") == "cnclip" else "OpenCLIP",
                    "dimension": int(vectors[index].shape[0]),
                    "path": f"embeddings/{seg['segment_id']}.npy",
                    "text_source": seg.get("text_source", "事件描述模板"),
                    "created_at": "",
                }
                for index, seg in enumerate(segments)
            ]
            self.segment_service.persist_segments(
                video_id=video_id,
                segments=segments,
                embeddings_meta=embeddings_meta,
            )
        except Exception:
            logger.exception("Segment pipeline failed for video_id=%s", video_id)
            return {"video_id": video_id, "built": False, "reason": "pipeline_failed"}

        self._mark_segment_built(video_id, True)
        return {
            "video_id": video_id,
            "built": True,
            "segment_count": len(segments),
            "event_segments": sum(1 for seg in segments if seg["events"]),
        }

    def _encode_segment_texts(self, clip_service, segments: List[Dict]):
        """逐段文本编码为 (N, dim) 矩阵。"""
        import numpy as np

        vectors = []
        for segment in segments:
            text = segment.get("text") or ""
            if not text.strip():
                raise ValueError(f"Empty segment text for {segment['segment_id']} (禁止空 text)")
            vectors.append(clip_service.encode_text(text))
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 事件元数据读写
    # ------------------------------------------------------------------ #

    def _load_events(self, video_id: str) -> Dict | None:
        path = self.settings.event_metadata_dir / f"{video_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load event metadata: %s", path)
            return None

    def _mark_segment_built(self, video_id: str, built: bool) -> bool:
        """在 events/{video_id}.json 写入 segment_built 标记（评审 #4 状态标记）。"""
        path = self.settings.event_metadata_dir / f"{video_id}.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["segment_built"] = built
            tmp = path.with_name(f"{path.name}.{id(payload)}.tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
            return True
        except Exception:
            logger.exception("Failed to mark segment_built=%s for video_id=%s", built, video_id)
            return False
