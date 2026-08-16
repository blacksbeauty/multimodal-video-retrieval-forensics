"""Regression tests for patent-logic fixes (2026-08-12).

Covers:
- P4: Event intent must give the event channel the highest weight, and a high
      event score must dominate the fused score (was: CLIP base + tiny boost).
- H2: Red-light plugin must reject a vehicle stopping on the stop line
      (legal) and only fire when the vehicle keeps moving after crossing.
- M3: The light must be *steadily* red around the crossing moment; a single
      red-frame blip is not enough.
"""

import unittest

from config import Settings
from core.schemas import DetectionVideoMetadata, TrajectoryVideoMetadata
from services.event_plugins.red_light_violation import RedLightViolation
from services.hybrid_search_service import HybridSearchService


def _detections() -> DetectionVideoMetadata:
    return DetectionVideoMetadata(
        video_id="video-1",
        video_name="traffic.mp4",
        video_path="videos/traffic.mp4",
        frames=[],
    )


def _trajectories(points, direction="left_to_right") -> TrajectoryVideoMetadata:
    payload_points = []
    for index, (x, y) in enumerate(points):
        payload_points.append(
            {
                "timestamp": float(index),
                "frame_path": f"frames/traffic_{index}.jpg",
                "bbox": [x - 2.0, y - 2.0, x + 2.0, y + 2.0],
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


def _red_light_plugin(states, **overrides):
    plugin = RedLightViolation()
    plugin._extract_traffic_light_states = lambda *args, **kwargs: states  # type: ignore[method-assign]
    config = {
        "line": [[12.5, 0.0], [12.5, 10.0]],
        "allowed_labels": ["car"],
        "min_displacement_px": 1.0,
        "state_window_sec": 2.0,
        "min_red_duration_sec": 1.5,
        "min_after_crossing_points": 2,
        "min_after_crossing_displacement_px": 15.0,
    }
    config.update(overrides)
    return plugin, config


class P4EventWeightTests(unittest.TestCase):
    def _service(self):
        settings = Settings()
        return HybridSearchService(
            settings=settings,
            query_rewrite_service=None,
            clip_search_service=None,
            ocr_search_service=None,
            result_aggregation_service=None,
        )

    def test_event_weights_dominant_for_event_intent(self) -> None:
        service = self._service()
        intent = {
            "kind": "event",
            "event_types": ["red_light_violation"],
            "event_confidence": 1.0,
            "attributes": {},
            "primary_entities": [],
        }
        weights = service.generate_dynamic_weights(intent)
        # P4: event channel must carry the highest weight for a pure event intent.
        self.assertGreaterEqual(weights["event"], 0.6)
        self.assertTrue(
            weights["event"] > weights["clip"]
            and weights["event"] > weights["detection"]
            and weights["event"] > weights["trajectory"]
            and weights["event"] > weights["ocr"]
        )

    def test_high_event_score_dominates_fused_score(self) -> None:
        service = self._service()
        intent = {
            "kind": "event",
            "event_types": ["red_light_violation"],
            "event_confidence": 1.0,
            "attributes": {},
            "primary_entities": [],
            "label_candidates": [],
            "direction": "",
        }
        item = {
            "clip_score": 0.10,
            "event_score": 0.90,
            "detection_score": 0.0,
            "trajectory_score": 0.0,
            "ocr_score": 0.0,
            "matched_by": ["event"],
        }
        fused = service._finalize_fused_item(item, intent)
        # Legacy formula gave ~0.235 (clip base + 0.15*event). Event-dominant
        # weighting must push the fused score well above 0.5.
        self.assertGreater(fused["score"], 0.5)


class H2RedLightCrossingTests(unittest.TestCase):
    def test_stopping_on_line_does_not_trigger_event(self) -> None:
        # Vehicle crosses the line (x=12.5) then barely moves (2px) — legal
        # red-light stop must NOT produce an event.
        plugin, config = _red_light_plugin(
            [
                {"timestamp": 1.0, "frame_path": "f/1.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 2.0, "frame_path": "f/2.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 3.0, "frame_path": "f/3.jpg", "state": "red", "confidence": 0.9},
            ]
        )
        trajectories = _trajectories([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (16.0, 5.0), (17.0, 5.0)])

        events = plugin.execute("video-1", _detections(), trajectories, config)

        self.assertEqual(events, [])

    def test_continuing_after_crossing_triggers_event(self) -> None:
        # Vehicle keeps moving (20px) after crossing — real red-light run.
        plugin, config = _red_light_plugin(
            [
                {"timestamp": 1.0, "frame_path": "f/1.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 2.0, "frame_path": "f/2.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 3.0, "frame_path": "f/3.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 4.0, "frame_path": "f/4.jpg", "state": "red", "confidence": 0.9},
            ]
        )
        trajectories = _trajectories([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (25.0, 5.0), (35.0, 5.0)])

        events = plugin.execute("video-1", _detections(), trajectories, config)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "red_light_violation")


class M3SustainedRedTests(unittest.TestCase):
    def test_single_red_blip_is_not_enough(self) -> None:
        # Only one red frame near the crossing; sustained-red check rejects it.
        plugin, config = _red_light_plugin(
            [
                {"timestamp": 1.0, "frame_path": "f/1.jpg", "state": "green", "confidence": 0.9},
                {"timestamp": 2.0, "frame_path": "f/2.jpg", "state": "green", "confidence": 0.9},
                {"timestamp": 2.9, "frame_path": "f/3.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 4.0, "frame_path": "f/4.jpg", "state": "green", "confidence": 0.9},
            ]
        )
        trajectories = _trajectories([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (25.0, 5.0), (35.0, 5.0)])

        events = plugin.execute("video-1", _detections(), trajectories, config)

        self.assertEqual(events, [])

    def test_sustained_red_triggers_event(self) -> None:
        # Red lasts >= 1.5s around the crossing time.
        plugin, config = _red_light_plugin(
            [
                {"timestamp": 1.5, "frame_path": "f/1.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 2.0, "frame_path": "f/2.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 2.5, "frame_path": "f/3.jpg", "state": "red", "confidence": 0.9},
                {"timestamp": 3.0, "frame_path": "f/4.jpg", "state": "red", "confidence": 0.9},
            ]
        )
        trajectories = _trajectories([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (25.0, 5.0), (35.0, 5.0)])

        events = plugin.execute("video-1", _detections(), trajectories, config)

        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
