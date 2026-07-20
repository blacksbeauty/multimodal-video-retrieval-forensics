import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Settings
from services.tracking_service import TrackingService


class _FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, _results):
        outputs = [
            np.asarray([[10.0, 10.0, 30.0, 30.0, 1.0, 0.9, 2.0, 0.0]], dtype=np.float32),
            np.asarray([[50.0, 10.0, 70.0, 30.0, 1.0, 0.85, 2.0, 0.0]], dtype=np.float32),
        ]
        value = outputs[self.calls] if self.calls < len(outputs) else np.empty((0, 8), dtype=np.float32)
        self.calls += 1
        return value


class TrackingServiceTests(unittest.TestCase):
    def test_process_video_metadata_generates_single_track_across_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            trajectory_dir = root / "metadata" / "trajectories"
            detection_dir.mkdir(parents=True, exist_ok=True)
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = detection_dir / "video-1.json"
            metadata_path.write_text(
                json.dumps(
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
                                        "bbox": [10.0, 10.0, 30.0, 30.0],
                                        "class_id": 2,
                                    }
                                ],
                            },
                            {
                                "video_name": "traffic.mp4",
                                "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                                "timestamp": 2.0,
                                "detections": [
                                    {
                                        "label": "car",
                                        "confidence": 0.85,
                                        "bbox": [20.0, 10.0, 40.0, 30.0],
                                        "class_id": 2,
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            settings = Settings(
                detection_metadata_dir=detection_dir,
                trajectory_metadata_dir=trajectory_dir,
                tracking_frame_rate=1,
            )
            service = TrackingService(settings)
            service.build_tracker = lambda frame_rate: _FakeTracker()

            metadata = service.process_video_metadata(metadata_path)

            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(len(metadata.tracks), 1)
            self.assertEqual(metadata.tracks[0].track_id, "video-1:1")
            self.assertEqual(metadata.tracks[0].frame_count, 2)
            self.assertEqual(metadata.tracks[0].label, "car")
            self.assertEqual(metadata.tracks[0].direction, "left_to_right")


if __name__ == "__main__":
    unittest.main()
