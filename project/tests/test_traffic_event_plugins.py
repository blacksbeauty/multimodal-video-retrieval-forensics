import unittest

from core.schemas import DetectionVideoMetadata, TrajectoryVideoMetadata
from services.event_plugins.red_light_violation import RedLightViolation
from services.event_plugins.vehicle_crosses_line import VehicleCrossesLine
from services.event_plugins.wrong_way_driving import WrongWayDriving
from services.query_rewrite_service import QueryRewriteService


def _detections() -> DetectionVideoMetadata:
    return DetectionVideoMetadata(
        video_id="video-1",
        video_name="traffic.mp4",
        video_path="videos/traffic.mp4",
        frames=[],
    )


def _trajectories(points, direction="left_to_right") -> TrajectoryVideoMetadata:
    payload_points = []
    for index, (x, y, bbox) in enumerate(points):
        payload_points.append(
            {
                "timestamp": float(index),
                "frame_path": f"frames/traffic_{index}.jpg",
                "bbox": bbox,
                "center_x": x,
                "center_y": y,
                "confidence": 0.9,
            }
        )
    return TrajectoryVideoMetadata.model_validate(
        {
            "video_id": "video-1",
            "video_name": "traffic.mp4",
            "video_path": "videos/traffic.mp4",
            "tracks": [
                {
                    "track_id": "video-1:1",
                    "label": "car",
                    "start_ts": 0.0,
                    "end_ts": float(max(len(payload_points) - 1, 0)),
                    "duration_sec": float(max(len(payload_points) - 1, 0)),
                    "frame_count": len(payload_points),
                    "avg_confidence": 0.9,
                    "max_confidence": 0.9,
                    "direction": direction,
                    "representative_frame": payload_points[-1]["frame_path"],
                    "points": payload_points,
                }
            ],
        }
    )


class TrafficEventPluginTests(unittest.TestCase):
    def test_line_plugin_uses_finite_segment_not_infinite_extension(self) -> None:
        plugin = VehicleCrossesLine()
        trajectories = _trajectories(
            [(0.0, 10.0, [0.0, 9.0, 2.0, 11.0]), (10.0, 10.0, [9.0, 9.0, 11.0, 11.0])]
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {"line": [[5.0, -2.0], [5.0, 2.0]], "allowed_labels": ["car"]},
        )

        self.assertEqual(events, [])

    def test_line_plugin_detects_vehicle_bbox_touching_line(self) -> None:
        plugin = VehicleCrossesLine()
        trajectories = _trajectories(
            [(6.0, 5.0, [4.0, 2.0, 8.0, 8.0]), (6.0, 5.0, [4.0, 2.0, 8.0, 8.0])]
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {"line": [[5.0, 0.0], [5.0, 10.0]], "allowed_labels": ["car"], "min_displacement_px": 0.0},
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "vehicle_crosses_line")
        self.assertEqual(events[0].attributes["crossing_mode"], "vehicle_bbox")

    def test_wrong_way_plugin_flags_motion_opposite_to_allowed_direction(self) -> None:
        plugin = WrongWayDriving()
        trajectories = _trajectories(
            [
                (100.0, 20.0, [95.0, 15.0, 105.0, 25.0]),
                (70.0, 20.0, [65.0, 15.0, 75.0, 25.0]),
                (40.0, 20.0, [35.0, 15.0, 45.0, 25.0]),
            ],
            direction="right_to_left",
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {
                "allowed_labels": ["car"],
                "allowed_direction": [1.0, 0.0],
                "min_track_points": 3,
                "min_duration_sec": 1.0,
                "min_displacement_px": 20.0,
                "max_direction_dot": 0.0,
            },
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "wrong_way_driving")
        self.assertLess(events[0].attributes["direction_dot"], 0.0)

    def test_wrong_way_plugin_ignores_motion_in_allowed_direction(self) -> None:
        plugin = WrongWayDriving()
        trajectories = _trajectories(
            [
                (40.0, 20.0, [35.0, 15.0, 45.0, 25.0]),
                (70.0, 20.0, [65.0, 15.0, 75.0, 25.0]),
                (100.0, 20.0, [95.0, 15.0, 105.0, 25.0]),
            ]
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {
                "allowed_labels": ["car"],
                "allowed_direction": [1.0, 0.0],
                "min_track_points": 3,
                "min_duration_sec": 1.0,
                "min_displacement_px": 20.0,
                "max_direction_dot": -0.3,
            },
        )

        self.assertEqual(events, [])

    def test_red_light_plugin_ignores_vehicle_crossing_during_green(self) -> None:
        plugin = RedLightViolation()
        plugin._extract_traffic_light_states = lambda *args, **kwargs: [  # type: ignore[method-assign]
            {"timestamp": 1.0, "frame_path": "frames/light.jpg", "state": "green", "confidence": 0.9}
        ]
        trajectories = _trajectories(
            [(0.0, 5.0, [0.0, 4.0, 2.0, 6.0]), (10.0, 5.0, [9.0, 4.0, 11.0, 6.0])]
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {
                "line": [[5.0, 0.0], [5.0, 10.0]],
                "allowed_labels": ["car"],
                "min_displacement_px": 1.0,
                "state_window_sec": 2.0,
            },
        )

        self.assertEqual(events, [])

    def test_query_rewrite_understands_all_chinese_traffic_events(self) -> None:
        service = QueryRewriteService()

        self.assertIn("vehicle_crosses_line", service.parse_query_intent("车辆压线").event_types)
        self.assertIn("wrong_way_driving", service.parse_query_intent("车辆逆行").event_types)
        self.assertIn("red_light_violation", service.parse_query_intent("车辆闯红灯").event_types)


class KeyFrameSnapshotTests(unittest.TestCase):
    """extract_three_keyframes 取证三帧快照测试（越线前/中/后）。"""

    def _points(self) -> list:
        from core.schemas import TrajectoryPoint

        # 轨迹：ts=0 在停止线上方 → ts=1 贴线（线上）→ ts=2 在停止线下方
        return [
            TrajectoryPoint(
                timestamp=0.0, frame_path="frames/traffic_0.jpg",
                bbox=[0.0, 0.0, 4.0, 4.0], center_x=2.0, center_y=2.0, confidence=0.9,
            ),
            TrajectoryPoint(
                timestamp=1.0, frame_path="frames/traffic_1.jpg",
                bbox=[4.0, 6.0, 6.0, 8.0], center_x=5.0, center_y=7.0, confidence=0.9,
            ),
            TrajectoryPoint(
                timestamp=2.0, frame_path="frames/traffic_2.jpg",
                bbox=[7.0, 11.0, 9.0, 13.0], center_x=8.0, center_y=12.0, confidence=0.9,
            ),
        ]

    def test_three_keyframes_before_crossing_after(self) -> None:
        plugin = VehicleCrossesLine()
        line = [[5.0, 5.0], [5.0, 10.0]]  # 竖直线段 x=5, y∈[5,10]
        evidence = ["frames/traffic_0.jpg", "frames/traffic_1.jpg", "frames/traffic_2.jpg"]

        snapshots = plugin.extract_three_keyframes(
            points=self._points(), evidence_frames=evidence, line=line, anchor_timestamp=1.0,
        )
        # A=越线前(上方最近) / B=越线中(贴线) / C=通过后(下方最近)
        self.assertEqual(
            snapshots,
            ["frames/traffic_0.jpg", "frames/traffic_1.jpg", "frames/traffic_2.jpg"],
        )

    def test_three_keyframes_without_line_uses_even_spacing(self) -> None:
        plugin = VehicleCrossesLine()
        evidence = [f"frames/traffic_{index}.jpg" for index in range(5)]

        snapshots = plugin.extract_three_keyframes(points=[], evidence_frames=evidence, line=None)
        # 无停止线：按证据帧时间均匀取首/中/尾
        self.assertEqual(
            snapshots,
            ["frames/traffic_0.jpg", "frames/traffic_2.jpg", "frames/traffic_4.jpg"],
        )

    def test_three_keyframes_empty_evidence_returns_empty(self) -> None:
        plugin = VehicleCrossesLine()
        self.assertEqual(
            plugin.extract_three_keyframes(points=[], evidence_frames=[], line=[[0, 0], [1, 1]]),
            [],
        )

    def test_three_keyframes_deduplicates_and_completes(self) -> None:
        plugin = VehicleCrossesLine()
        # 全部点都在线上方（无 B/C 候选）→ 兜底：B 取证据中位、C 取证据最大
        from core.schemas import TrajectoryPoint

        points = [
            TrajectoryPoint(
                timestamp=0.0, frame_path="frames/traffic_0.jpg",
                bbox=[0.0, 0.0, 2.0, 2.0], center_x=1.0, center_y=1.0, confidence=0.9,
            ),
            TrajectoryPoint(
                timestamp=2.0, frame_path="frames/traffic_2.jpg",
                bbox=[0.0, 0.0, 2.0, 2.0], center_x=1.0, center_y=1.0, confidence=0.9,
            ),
        ]
        evidence = [f"frames/traffic_{index}.jpg" for index in range(5)]
        snapshots = plugin.extract_three_keyframes(
            points=points, evidence_frames=evidence, line=[[5.0, 5.0], [5.0, 10.0]], anchor_timestamp=3.0,
        )
        # A=上方最近帧；B/C 无候选 → evidence 中位/最大；去重后补足到 3 帧
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(len(set(snapshots)), 3)

    def test_plugin_event_contains_key_snapshots(self) -> None:
        plugin = VehicleCrossesLine()
        trajectories = _trajectories(
            [
                (6.0, 5.0, [4.0, 2.0, 8.0, 8.0]),
                (6.0, 5.0, [4.0, 2.0, 8.0, 8.0]),
                (10.0, 12.0, [8.0, 9.0, 12.0, 15.0]),
            ]
        )

        events = plugin.execute(
            "video-1",
            _detections(),
            trajectories,
            {"line": [[5.0, 0.0], [5.0, 10.0]], "allowed_labels": ["car"], "min_displacement_px": 0.0},
        )
        self.assertEqual(len(events), 1)
        snapshots = events[0].key_snapshots
        # 快照帧应属于"事件证据帧 ∪ 轨迹点帧"（Frame_C 可取自轨迹点、未必在 evidence 内）
        valid_frames = set(events[0].evidence_frames)
        for track in trajectories.tracks:
            valid_frames.update(point.frame_path for point in track.points)
        self.assertLessEqual(len(snapshots), 3)
        self.assertTrue(all(path in valid_frames for path in snapshots))


class RedLightClassifyTests(unittest.TestCase):
    """_classify_traffic_light_state 判色鲁棒性（整图比例 + 亮度重心，远距小灯可判）。"""

    def _make_lamp_image(self, color_bgr, y_center):
        import cv2
        import numpy as np

        image = np.full((80, 40, 3), 60, dtype=np.uint8)  # 深灰背景（非亮灯）
        cv2.circle(image, (20, y_center), 5, tuple(color_bgr), -1)
        return image

    def _classify(self, color_bgr, y_center, min_state_score=0.005):
        import cv2
        import tempfile
        from pathlib import Path

        plugin = RedLightViolation()
        image = self._make_lamp_image(color_bgr, y_center)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lamp.png"
            cv2.imwrite(str(path), image)
            return plugin._classify_traffic_light_state(str(path), [0, 0, 40, 80], min_state_score)

    def test_red_lamp_at_top(self):
        result = self._classify((0, 0, 255), y_center=12)
        self.assertIsNotNone(result)
        self.assertEqual(result["state"], "red")

    def test_green_lamp_at_bottom(self):
        result = self._classify((0, 255, 0), y_center=68)
        self.assertIsNotNone(result)
        self.assertEqual(result["state"], "green")

    def test_yellow_lamp_at_middle(self):
        result = self._classify((0, 255, 255), y_center=40)
        self.assertIsNotNone(result)
        self.assertEqual(result["state"], "yellow")

    def test_dark_region_returns_none(self):
        result = self._classify((30, 30, 30), y_center=40)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
