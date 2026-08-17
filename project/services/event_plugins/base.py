from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence

from core.schemas import (
    DetectionVideoMetadata,
    EventMetadata,
    TrajectoryPoint,
    TrajectoryVideoMetadata,
)
from services.event_plugins.geometry import closest_point_on_segment, point_to_segment_distance

# 检测类别 → 中文（用于事件 description / segment.text，数据规范 v1.1 §3.7/§3.8）
LABEL_CN: Dict[str, str] = {
    "car": "小汽车",
    "truck": "货车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "person": "行人",
    "traffic light": "信号灯",
    "bicycle": "自行车",
    "van": "面包车",
}


def label_to_chinese(label: str) -> str:
    """S7: 检测类别英文 label → 中文（未知类别回退原文）。"""
    return LABEL_CN.get(str(label).strip().casefold(), str(label))


class EventPluginBase(ABC):
    """Abstract base class for all rule-based traffic event plugins."""

    plugin_name: str = ""
    event_type: str = ""

    @abstractmethod
    def execute(
        self,
        video_id: str,
        detections: DetectionVideoMetadata,
        trajectories: TrajectoryVideoMetadata,
        config: Dict[str, Any],
    ) -> List[EventMetadata]:
        """Produce traffic events for one video."""

    def pick_representative_frame(
        self,
        evidence_frames: Sequence[str],
        fallback: str = "",
    ) -> str:
        """Code Review Must Fix：事件代表帧必须来自证据帧列表（事件时间窗内）。

        旧实现直接用 ``track.representative_frame``（轨迹级代表帧），该帧可能在
        事件时间窗之外、甚至属于另一辆车（例：闯红灯事件窗 [14.8, 15.8] 的代表帧
        被选到 18.2s 的绿灯+另一辆车画面），检索页面会用错误帧误导评审。

        规则：优先取证据帧列表第一帧（最早的事件证据帧）；空列表回退 fallback。
        """
        if evidence_frames:
            return str(evidence_frames[0])
        return str(fallback or "")

    # ------------------------------------------------------------------ #
    # 取证三帧快照（越线前 / 越线中 / 通过后）
    # ------------------------------------------------------------------ #

    def extract_three_keyframes(
        self,
        points: Sequence[TrajectoryPoint],
        evidence_frames: Sequence[str],
        line: Sequence[Sequence[float]] | None,
        anchor_timestamp: float | None = None,
    ) -> List[str]:
        """按交通违法取证规范返回三帧快照 [越线前, 越线中, 通过后] 的帧路径。

        参数:
          points: 轨迹点列表（TrajectoryPoint 自带 frame_path/timestamp/center_x/center_y，
                  取点即取帧，无需时间戳匹配）。
          evidence_frames: 事件证据帧路径（List[str]，仅路径）。
          line: 停止线线段端点 [[x1,y1],[x2,y2]]（config "line" 字段）；None 时
                跳过几何筛选，按证据帧时间均匀取首/中/尾（wrong_way 等无停止线场景）。
          anchor_timestamp: 越线时刻（crossing timestamp）；None 时取证据帧时间中位数。

        筛选规则（优先级，逐帧兜底）:
          Frame_A 越线前: center_y < 线上投影点_y 且 ts <= anchor，取离线段最近点；
          Frame_B 越线中: 点到线段距离 <= 5px，取最近点；
          Frame_C 通过后: center_y > 线上投影点_y 且 ts >= anchor，取离线段最近点。
        结果去重后不足 3 帧时按证据帧时间序补足；evidence_frames 为空返回 []。
        """
        if not evidence_frames:
            return []

        evidence_sorted = sorted(evidence_frames, key=self._frame_timestamp)
        if not line or len(line) != 2 or any(len(p) != 2 for p in line):
            # 无停止线：按证据帧时间均匀取首/中/尾（取证三帧降级方案）。
            return self._evenly_spaced_frames(evidence_sorted)

        x1, y1 = map(float, line[0])
        x2, y2 = map(float, line[1])
        anchor = (
            float(anchor_timestamp)
            if anchor_timestamp is not None
            else self._frame_timestamp(evidence_sorted[len(evidence_sorted) // 2])
        )

        # 逐点计算：与停止线的距离、投影点 y 相对位置（负=上方，正=下方）
        above, crossing, below = [], [], []
        for point in points:
            try:
                cx, cy = float(point.center_x), float(point.center_y)
                ts = float(point.timestamp)
                dist = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
                _, proj_y = closest_point_on_segment(cx, cy, x1, y1, x2, y2)
            except (TypeError, ValueError):
                continue
            if dist <= 5.0:
                crossing.append((dist, ts, str(point.frame_path)))
            elif cy < proj_y and ts <= anchor:
                above.append((dist, ts, str(point.frame_path)))
            elif cy > proj_y and ts >= anchor:
                below.append((dist, ts, str(point.frame_path)))

        def nearest(candidates: Sequence[tuple[float, float, str]], fallback_index: int) -> str:
            if candidates:
                return min(candidates, key=lambda item: item[0])[2]
            return evidence_sorted[fallback_index]

        frame_a = nearest(above, 0)                 # 越线前：最接近线、在上方
        frame_b = nearest(crossing, len(evidence_sorted) // 2)  # 越线中：最近/贴线
        frame_c = nearest(below, len(evidence_sorted) - 1)      # 通过后：最接近线、在下方

        return self._dedupe_and_complete([frame_a, frame_b, frame_c], evidence_sorted)

    def _frame_timestamp(self, frame_path: str) -> float:
        """从帧文件名解析时间戳（{video_name}_{ts:.2f}.jpg）；失败回退 0.0。"""
        stem = Path(str(frame_path)).stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                return float(parts[1])
            except ValueError:
                pass
        return 0.0

    def _evenly_spaced_frames(self, evidence_sorted: Sequence[str]) -> List[str]:
        """按时间均匀取首/中/尾三帧（无停止线时的降级方案）。"""
        count = len(evidence_sorted)
        if count <= 3:
            return list(evidence_sorted)
        return [
            evidence_sorted[0],
            evidence_sorted[count // 2],
            evidence_sorted[count - 1],
        ]

    def _dedupe_and_complete(
        self,
        frames: Sequence[str],
        evidence_sorted: Sequence[str],
    ) -> List[str]:
        """去重并保持顺序；不足 3 帧时按证据帧时间序补齐。"""
        unique: List[str] = []
        for frame in frames:
            if frame and frame not in unique:
                unique.append(frame)
        for frame in evidence_sorted:
            if len(unique) >= 3:
                break
            if frame not in unique:
                unique.append(frame)
        return unique
