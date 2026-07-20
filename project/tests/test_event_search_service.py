import tempfile
import unittest
from pathlib import Path

from config import Settings
from services.event_search_service import EventSearchService
from services.query_rewrite_service import QueryRewriteService


class EventSearchServiceTests(unittest.TestCase):
    def test_search_finds_vehicle_crosses_line_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "metadata" / "events"
            event_dir.mkdir(parents=True, exist_ok=True)
            event_file = event_dir / "video-1.json"
            event_file.write_text(
                """
                {
                  "video_id": "video-1",
                  "video_name": "traffic.mp4",
                  "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                  "events": [
                    {
                      "event_id": "video-1:vehicle_crosses_line:video-1:1",
                      "event_type": "vehicle_crosses_line",
                      "plugin_name": "vehicle_crosses_line",
                      "video_id": "video-1",
                      "video_name": "traffic.mp4",
                      "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                      "start_ts": 1.0,
                      "end_ts": 3.0,
                      "track_ids": ["video-1:1"],
                      "confidence": 0.9,
                      "representative_frame": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                      "evidence_frames": [],
                      "attributes": {},
                      "description": "car crosses configured line"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            settings = Settings(event_metadata_dir=event_dir)
            service = EventSearchService(settings, QueryRewriteService())
            results = service.search("vehicle_crosses_line", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].event_type, "vehicle_crosses_line")
            self.assertEqual(results[0].track_ids, ["video-1:1"])

    def test_search_understands_red_light_violation_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "metadata" / "events"
            event_dir.mkdir(parents=True, exist_ok=True)
            event_file = event_dir / "video-1.json"
            event_file.write_text(
                """
                {
                  "video_id": "video-1",
                  "video_name": "traffic.mp4",
                  "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                  "events": [
                    {
                      "event_id": "video-1:red_light_violation:video-1:1",
                      "event_type": "red_light_violation",
                      "plugin_name": "red_light_violation",
                      "video_id": "video-1",
                      "video_name": "traffic.mp4",
                      "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                      "start_ts": 1.0,
                      "end_ts": 2.0,
                      "track_ids": ["video-1:1"],
                      "confidence": 0.95,
                      "representative_frame": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                      "evidence_frames": [],
                      "attributes": {
                        "light_state": "red"
                      },
                      "description": "car crossed stop line during red light"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            settings = Settings(event_metadata_dir=event_dir)
            service = EventSearchService(settings, QueryRewriteService())
            results = service.search("\u7ea2\u706f\u65f6\u7a7f\u8fc7\u505c\u6b62\u7ebf\u7684\u6c7d\u8f66", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].event_type, "red_light_violation")


if __name__ == "__main__":
    unittest.main()
