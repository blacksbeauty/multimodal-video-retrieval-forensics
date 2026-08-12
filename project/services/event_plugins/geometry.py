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
