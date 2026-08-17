from __future__ import annotations

from math import hypot
from typing import Any, Sequence


Point = tuple[float, float]
Line = Sequence[Sequence[float]]


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point, epsilon: float = 1e-6) -> bool:
    return (
        min(a[0], c[0]) - epsilon <= b[0] <= max(a[0], c[0]) + epsilon
        and min(a[1], c[1]) - epsilon <= b[1] <= max(a[1], c[1]) + epsilon
    )


def segments_intersect(first: Sequence[float], second: Sequence[float], third: Sequence[float], fourth: Sequence[float]) -> bool:
    """Return whether two finite 2D line segments touch or cross."""
    a, b, c, d = _point(first), _point(second), _point(third), _point(fourth)
    epsilon = 1e-6
    values = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if all(abs(value) <= epsilon for value in values):
        return _on_segment(a, c, b) or _on_segment(a, d, b) or _on_segment(c, a, d) or _on_segment(c, b, d)
    return (
        (values[0] >= -epsilon and values[1] <= epsilon or values[0] <= epsilon and values[1] >= -epsilon)
        and (values[2] >= -epsilon and values[3] <= epsilon or values[2] <= epsilon and values[3] >= -epsilon)
    )


def closest_point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """返回点 (px, py) 在线段 (x1,y1)-(x2,y2) 上的最近投影点坐标。

    三帧取证快照（extract_three_keyframes）用它判定车辆质心相对停止线的
    上下位置（以投影点 y 为界，兼容竖直/倾斜线段）。
    """
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:  # 退化线段（两点重合）
        return float(x1), float(y1)
    t = ((float(px) - float(x1)) * dx + (float(py) - float(y1)) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return float(x1) + t * dx, float(y1) + t * dy


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """返回点 (px, py) 到线段 (x1,y1)-(x2,y2) 的最短距离（欧氏）。"""
    cx, cy = closest_point_on_segment(px, py, x1, y1, x2, y2)
    return hypot(float(px) - cx, float(py) - cy)


def point_in_bbox(point: Sequence[float], bbox: Sequence[float], epsilon: float = 1e-6) -> bool:
    if len(bbox) != 4:
        return False
    x, y = _point(point)
    x1, y1, x2, y2 = map(float, bbox)
    return min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon and min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon


def bbox_touches_line(bbox: Sequence[float], line: Line) -> bool:
    """Return whether an axis-aligned vehicle box touches a finite line segment."""
    if len(bbox) != 4 or len(line) != 2 or any(len(point) != 2 for point in line):
        return False
    x1, y1, x2, y2 = map(float, bbox)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    line_start, line_end = line[0], line[1]
    if point_in_bbox(line_start, bbox) or point_in_bbox(line_end, bbox):
        return True
    return any(segments_intersect(corners[index], corners[(index + 1) % 4], line_start, line_end) for index in range(4))


def trajectory_displacement(points: Sequence[Any]) -> float:
    if len(points) < 2:
        return 0.0
    return hypot(float(points[-1].center_x) - float(points[0].center_x), float(points[-1].center_y) - float(points[0].center_y))


def trajectory_main_direction(
    points: Sequence[Any],
    start_frac: float = 0.25,
    end_frac: float = 0.75,
) -> tuple[float, float] | None:
    """Robust trajectory direction using the middle segment of the track.

    Uses the displacement between the ``start_frac`` and ``end_frac``
    percentile points instead of the raw first/last points, which makes the
    direction estimate resistant to jitter at the track head/tail and to
    mid-track U-turns (the overall motion vector stays representative).
    """
    if len(points) < 2:
        return None
    last_index = len(points) - 1
    start_index = int(round(last_index * max(min(start_frac, 1.0), 0.0)))
    end_index = int(round(last_index * max(min(end_frac, 1.0), 0.0)))
    start_index = max(min(start_index, last_index), 0)
    end_index = max(min(end_index, last_index), 0)
    if end_index <= start_index:
        end_index = min(start_index + 1, last_index)
    start, end = points[start_index], points[end_index]
    return normalize_vector((float(end.center_x) - float(start.center_x), float(end.center_y) - float(start.center_y)))


def find_line_contact(points: Sequence[Any], line: Line, min_displacement_px: float = 1.0) -> dict[str, float | str] | None:
    """Find the first finite-line contact using vehicle boxes or center motion."""
    if len(points) < 2 or trajectory_displacement(points) < max(float(min_displacement_px), 0.0):
        return None

    for index, point in enumerate(points):
        if bbox_touches_line(point.bbox, line):
            return {"index": float(index), "timestamp": float(point.timestamp), "mode": "vehicle_bbox"}

    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        if segments_intersect(
            (previous.center_x, previous.center_y),
            (current.center_x, current.center_y),
            line[0],
            line[1],
        ):
            return {"index": float(index), "timestamp": float(current.timestamp), "mode": "trajectory_center"}
    return None


def normalize_vector(vector: Sequence[float]) -> tuple[float, float] | None:
    if len(vector) != 2:
        return None
    length = hypot(float(vector[0]), float(vector[1]))
    if length <= 1e-9:
        return None
    return float(vector[0]) / length, float(vector[1]) / length


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return whether a point is inside a simple polygon using ray casting."""
    if len(polygon) < 3:
        return False
    x, y = _point(point)
    inside = False
    for index in range(len(polygon)):
        x1, y1 = _point(polygon[index])
        x2, y2 = _point(polygon[(index + 1) % len(polygon)])
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside
