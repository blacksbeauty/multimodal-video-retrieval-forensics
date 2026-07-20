import tempfile
import unittest
from pathlib import Path

from config import Settings
from services.query_rewrite_service import QueryRewriteService
from services.trajectory_search_service import TrajectorySearchService


class TrajectorySearchServiceTests(unittest.TestCase):
    def _build_service(self, root: Path) -> TrajectorySearchService:
        trajectory_dir = root / "metadata" / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(trajectory_metadata_dir=trajectory_dir)
        return TrajectorySearchService(settings, QueryRewriteService())

    def test_search_tracks_filters_by_label_and_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory_dir = root / "metadata" / "trajectories"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_file = trajectory_dir / "video-1.json"
            trajectory_file.write_text(
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
                      "points": []
                    },
                    {
                      "track_id": "video-1:2",
                      "label": "person",
                      "start_ts": 4.0,
                      "end_ts": 5.0,
                      "duration_sec": 1.0,
                      "frame_count": 2,
                      "avg_confidence": 0.7,
                      "max_confidence": 0.8,
                      "direction": "right_to_left",
                      "representative_frame": "d:/all-seeing vision/rag/project/frames/traffic_4.0.jpg",
                      "points": []
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            car_results = service.search_tracks("car", top_k=5)
            person_results = service.search_tracks("person", top_k=5, direction="right_to_left")
            blocked_results = service.search_tracks("person", top_k=5, direction="left_to_right")

            self.assertEqual(len(car_results), 1)
            self.assertEqual(car_results[0].track_id, "video-1:1")
            self.assertEqual(len(person_results), 1)
            self.assertEqual(person_results[0].direction, "right_to_left")
            self.assertEqual(blocked_results, [])

    def test_search_as_frames_returns_hybrid_ready_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory_dir = root / "metadata" / "trajectories"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_file = trajectory_dir / "video-1.json"
            trajectory_file.write_text(
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
                      "points": []
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            results = service.search_as_frames("car", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["video_id"], "video-1")
            self.assertEqual(results[0]["trajectory_score"], 0.9)
            self.assertEqual(results[0]["matched_by"], ["trajectory"])
            self.assertEqual(results[0]["matched_direction"], "left_to_right")

    def test_search_tracks_supports_chinese_motion_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory_dir = root / "metadata" / "trajectories"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_file = trajectory_dir / "video-1.json"
            trajectory_file.write_text(
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
                      "points": []
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            results = service.search_tracks("\u6c7d\u8f66\u4ece\u5de6\u5230\u53f3", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].label, "car")
            self.assertEqual(results[0].direction, "left_to_right")


if __name__ == "__main__":
    unittest.main()
