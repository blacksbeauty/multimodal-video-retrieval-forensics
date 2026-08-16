import tempfile
import unittest
from pathlib import Path

from config import Settings
from core.schemas import DetectionVideoMetadata, TrajectoryVideoMetadata
from services.event_service import EventService
from PIL import Image, ImageDraw


class EventServiceTests(unittest.TestCase):
    def _write_detection_metadata(self, path: Path) -> None:
        path.write_text(
            """
            {
              "video_id": "video-1",
              "video_name": "traffic.mp4",
              "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
              "frames": [
                {
                  "video_name": "traffic.mp4",
                  "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_1.0.jpg",
                  "timestamp": 1.0,
                  "detections": [
                    {
                      "label": "car",
                      "confidence": 0.9,
                      "bbox": [10.0, 10.0, 20.0, 20.0],
                      "class_id": 2
                    }
                  ]
                }
              ]
            }
            """,
            encoding="utf-8",
        )

    def _write_trajectory_metadata(self, path: Path) -> None:
        path.write_text(
            """
            {
              "video_id": "video-1",
              "video_name": "traffic.mp4",
              "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
              "tracks": [
                {
                  "track_id": "video-1:1",
                  "label": "car",
                  "start_ts": 1.0,
                  "end_ts": 3.0,
                  "duration_sec": 2.0,
                  "frame_count": 3,
                  "avg_confidence": 0.9,
                  "max_confidence": 0.95,
                  "direction": "left_to_right",
                  "representative_frame": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                  "points": [
                    {
                      "timestamp": 1.0,
                      "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_1.0.jpg",
                      "bbox": [10.0, 10.0, 20.0, 20.0],
                      "center_x": 10.0,
                      "center_y": 10.0,
                      "confidence": 0.9
                    },
                    {
                      "timestamp": 2.0,
                      "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                      "bbox": [50.0, 10.0, 60.0, 20.0],
                      "center_x": 50.0,
                      "center_y": 10.0,
                      "confidence": 0.9
                    },
                    {
                      "timestamp": 3.0,
                      "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_3.0.jpg",
                      "bbox": [90.0, 10.0, 100.0, 20.0],
                      "center_x": 90.0,
                      "center_y": 10.0,
                      "confidence": 0.9
                    }
                  ]
                }
              ]
            }
            """,
            encoding="utf-8",
        )

    def test_process_video_metadata_generates_vehicle_crosses_line_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            trajectory_dir = root / "metadata" / "trajectories"
            event_dir = root / "metadata" / "events"
            config_dir = root / "configs" / "events"
            detection_dir.mkdir(parents=True, exist_ok=True)
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            event_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)

            detection_path = detection_dir / "video-1.json"
            trajectory_path = trajectory_dir / "video-1.json"
            self._write_detection_metadata(detection_path)
            self._write_trajectory_metadata(trajectory_path)
            (config_dir / "vehicle_crosses_line.json").write_text(
                """
                {
                  "line": [
                    [40.0, 0.0],
                    [40.0, 30.0]
                  ],
                  "allowed_labels": ["car"]
                }
                """,
                encoding="utf-8",
            )
            (config_dir / "red_light_violation.json").write_text('{"enabled": false}', encoding="utf-8")

            settings = Settings(
                detection_metadata_dir=detection_dir,
                trajectory_metadata_dir=trajectory_dir,
                event_metadata_dir=event_dir,
                event_config_dir=config_dir,
                event_plugin_names=["vehicle_crosses_line"],
            )
            service = EventService(settings)

            bundle = service.process_video_metadata(trajectory_path, detection_dir=detection_dir, plugin_names=["vehicle_crosses_line"])

            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(len(bundle.events), 1)
            self.assertEqual(bundle.events[0].event_type, "vehicle_crosses_line")
            self.assertEqual(bundle.events[0].track_ids, ["video-1:1"])

    def test_red_light_violation_plugin_generates_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            trajectory_dir = root / "metadata" / "trajectories"
            event_dir = root / "metadata" / "events"
            config_dir = root / "configs" / "events"
            frames_dir = root / "frames"
            detection_dir.mkdir(parents=True, exist_ok=True)
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            event_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Four frames: the H2 anti-false-positive fix in the plugin
            # requires >= min_after_crossing_points trajectory points AFTER
            # the crossing, proving the vehicle kept moving past the stop line
            # (a track whose crossing lands on the last point is deliberately
            # treated as "stopping on the line", not a violation).
            for index in (1, 2, 3, 4):
                image = Image.new("RGB", (120, 120), "black")
                draw = ImageDraw.Draw(image)
                draw.rectangle((10, 10, 40, 40), fill=(255, 0, 0))
                image.save(frames_dir / f"traffic_{index}.jpg")

            detection_path = detection_dir / "video-1.json"
            detection_path.write_text(
                f"""
                {{
                  "video_id": "video-1",
                  "video_name": "traffic.mp4",
                  "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                  "frames": [
                    {{
                      "video_name": "traffic.mp4",
                      "frame_path": "{(frames_dir / 'traffic_1.jpg').as_posix()}",
                      "timestamp": 1.0,
                      "detections": [
                        {{
                          "label": "traffic light",
                          "confidence": 0.95,
                          "bbox": [10.0, 10.0, 40.0, 40.0],
                          "class_id": 9
                        }},
                        {{
                          "label": "car",
                          "confidence": 0.9,
                          "bbox": [10.0, 60.0, 30.0, 80.0],
                          "class_id": 2
                        }}
                      ]
                    }},
                    {{
                      "video_name": "traffic.mp4",
                      "frame_path": "{(frames_dir / 'traffic_2.jpg').as_posix()}",
                      "timestamp": 2.0,
                      "detections": [
                        {{
                          "label": "traffic light",
                          "confidence": 0.95,
                          "bbox": [10.0, 10.0, 40.0, 40.0],
                          "class_id": 9
                        }},
                        {{
                          "label": "car",
                          "confidence": 0.9,
                          "bbox": [60.0, 60.0, 80.0, 80.0],
                          "class_id": 2
                        }}
                      ]
                    }},
                    {{
                      "video_name": "traffic.mp4",
                      "frame_path": "{(frames_dir / 'traffic_3.jpg').as_posix()}",
                      "timestamp": 3.0,
                      "detections": [
                        {{
                          "label": "traffic light",
                          "confidence": 0.95,
                          "bbox": [10.0, 10.0, 40.0, 40.0],
                          "class_id": 9
                        }},
                        {{
                          "label": "car",
                          "confidence": 0.9,
                          "bbox": [70.0, 60.0, 90.0, 80.0],
                          "class_id": 2
                        }}
                      ]
                    }},
                    {{
                      "video_name": "traffic.mp4",
                      "frame_path": "{(frames_dir / 'traffic_4.jpg').as_posix()}",
                      "timestamp": 4.0,
                      "detections": [
                        {{
                          "label": "traffic light",
                          "confidence": 0.95,
                          "bbox": [10.0, 10.0, 40.0, 40.0],
                          "class_id": 9
                        }},
                        {{
                          "label": "car",
                          "confidence": 0.9,
                          "bbox": [95.0, 60.0, 115.0, 80.0],
                          "class_id": 2
                        }}
                      ]
                    }}
                  ]
                }}
                """,
                encoding="utf-8",
            )
            trajectory_path = trajectory_dir / "video-1.json"
            trajectory_path.write_text(
                f"""
                {{
                  "video_id": "video-1",
                  "video_name": "traffic.mp4",
                  "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                  "tracks": [
                    {{
                      "track_id": "video-1:1",
                      "label": "car",
                      "start_ts": 1.0,
                      "end_ts": 4.0,
                      "duration_sec": 3.0,
                      "frame_count": 4,
                      "avg_confidence": 0.9,
                      "max_confidence": 0.9,
                      "direction": "left_to_right",
                      "representative_frame": "{(frames_dir / 'traffic_4.jpg').as_posix()}",
                      "points": [
                        {{
                          "timestamp": 1.0,
                          "frame_path": "{(frames_dir / 'traffic_1.jpg').as_posix()}",
                          "bbox": [10.0, 60.0, 30.0, 80.0],
                          "center_x": 20.0,
                          "center_y": 70.0,
                          "confidence": 0.9
                        }},
                        {{
                          "timestamp": 2.0,
                          "frame_path": "{(frames_dir / 'traffic_2.jpg').as_posix()}",
                          "bbox": [60.0, 60.0, 80.0, 80.0],
                          "center_x": 70.0,
                          "center_y": 70.0,
                          "confidence": 0.9
                        }},
                        {{
                          "timestamp": 3.0,
                          "frame_path": "{(frames_dir / 'traffic_3.jpg').as_posix()}",
                          "bbox": [70.0, 60.0, 90.0, 80.0],
                          "center_x": 90.0,
                          "center_y": 70.0,
                          "confidence": 0.9
                        }},
                        {{
                          "timestamp": 4.0,
                          "frame_path": "{(frames_dir / 'traffic_4.jpg').as_posix()}",
                          "bbox": [95.0, 60.0, 115.0, 80.0],
                          "center_x": 115.0,
                          "center_y": 70.0,
                          "confidence": 0.9
                        }}
                      ]
                    }}
                  ]
                }}
                """,
                encoding="utf-8",
            )
            (config_dir / "vehicle_crosses_line.json").write_text(
                '{"line": [[40.0, 0.0], [40.0, 120.0]], "allowed_labels": ["car"]}',
                encoding="utf-8",
            )
            (config_dir / "red_light_violation.json").write_text(
                '{"enabled": true, "line": [[40.0, 0.0], [40.0, 120.0]], "allowed_labels": ["car"], "state_window_sec": 2.0, "traffic_light_min_confidence": 0.25, "traffic_light_min_state_score": 0.01}',
                encoding="utf-8",
            )

            settings = Settings(
                detection_metadata_dir=detection_dir,
                trajectory_metadata_dir=trajectory_dir,
                event_metadata_dir=event_dir,
                event_config_dir=config_dir,
                event_plugin_names=["red_light_violation"],
            )
            service = EventService(settings)

            bundle = service.process_video_metadata(trajectory_path, detection_dir=detection_dir, plugin_names=["red_light_violation"])

            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(len(bundle.events), 1)
            self.assertEqual(bundle.events[0].event_type, "red_light_violation")
            self.assertEqual(bundle.events[0].attributes["light_state"], "red")
            self.assertEqual(bundle.events[0].track_ids, ["video-1:1"])


if __name__ == "__main__":
    unittest.main()
