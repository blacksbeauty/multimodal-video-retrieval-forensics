import tempfile
import unittest
from pathlib import Path

from config import Settings
from services.detection_search_service import DetectionSearchService
from services.query_rewrite_service import QueryRewriteService


class DetectionSearchServiceTests(unittest.TestCase):
    def _build_service(self, root: Path) -> DetectionSearchService:
        detection_dir = root / "metadata" / "detections"
        detection_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(detection_metadata_dir=detection_dir)
        return DetectionSearchService(settings, QueryRewriteService())

    def test_search_objects_supports_chinese_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            detection_dir.mkdir(parents=True, exist_ok=True)
            detection_file = detection_dir / "video-1.json"
            detection_file.write_text(
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
                          "bbox": [1.0, 2.0, 3.0, 4.0],
                          "class_id": 2
                        },
                        {
                          "label": "traffic light",
                          "confidence": 0.8,
                          "bbox": [5.0, 6.0, 7.0, 8.0],
                          "class_id": 9
                        }
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            car_results = service.search_objects("\u6c7d\u8f66", top_k=5)
            light_results = service.search_objects("\u7ea2\u7eff\u706f", top_k=5)

            self.assertEqual(len(car_results), 1)
            self.assertEqual(car_results[0].matched_label, "car")
            self.assertEqual(len(light_results), 1)
            self.assertEqual(light_results[0].matched_label, "traffic light")

    def test_search_objects_supports_phrase_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            detection_dir.mkdir(parents=True, exist_ok=True)
            detection_file = detection_dir / "video-1.json"
            detection_file.write_text(
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
                          "bbox": [1.0, 2.0, 3.0, 4.0],
                          "class_id": 2
                        }
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            results = service.search_objects("\u6c7d\u8f66\u4ece\u5de6\u5230\u53f3", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].matched_label, "car")

    def test_search_objects_supports_relational_frame_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detection_dir = root / "metadata" / "detections"
            detection_dir.mkdir(parents=True, exist_ok=True)
            detection_file = detection_dir / "video-1.json"
            detection_file.write_text(
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
                          "label": "traffic light",
                          "confidence": 0.8,
                          "bbox": [100.0, 100.0, 120.0, 140.0],
                          "class_id": 9
                        },
                        {
                          "label": "car",
                          "confidence": 0.9,
                          "bbox": [110.0, 140.0, 180.0, 200.0],
                          "class_id": 2
                        }
                      ]
                    },
                    {
                      "video_name": "traffic.mp4",
                      "frame_path": "d:/all-seeing vision/rag/project/frames/traffic_2.0.jpg",
                      "timestamp": 2.0,
                      "detections": [
                        {
                          "label": "car",
                          "confidence": 0.95,
                          "bbox": [600.0, 140.0, 680.0, 200.0],
                          "class_id": 2
                        }
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            service = self._build_service(root)
            results = service.search_objects("\u7ea2\u7eff\u706f\u9644\u8fd1\u7684\u8f66", top_k=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].timestamp, 1.0)
            self.assertEqual(results[0].matched_label, "car")


if __name__ == "__main__":
    unittest.main()
