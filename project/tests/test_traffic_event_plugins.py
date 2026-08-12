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


if __name__ == "__main__":
    unittest.main()
