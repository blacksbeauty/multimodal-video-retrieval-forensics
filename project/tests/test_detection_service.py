import tempfile
import unittest
from pathlib import Path

from config import Settings
from core.schemas import DetectionFrameMetadata, DetectionVideoMetadata
from services.detection_service import DetectionService


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    def __init__(self, cls_values, conf_values, bbox_values):
        self.cls = _FakeTensor(cls_values)
        self.conf = _FakeTensor(conf_values)
        self.xyxy = _FakeTensor(bbox_values)


class _FakeResult:
    def __init__(self, names, cls_values, conf_values, bbox_values):
        self.names = names
        self.boxes = _FakeBoxes(cls_values, conf_values, bbox_values)


class DetectionServicePathTests(unittest.TestCase):
    def test_prefers_project_frames_dir_for_metadata_frame_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir = root / "metadata" / "detections"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            canonical_frame = frames_dir / "traffic_video_modified_0.0.jpg"
            canonical_frame.write_bytes(b"canonical")

            copied_dir = root / "copied"
            copied_dir.mkdir(parents=True, exist_ok=True)
            copied_frame = copied_dir / canonical_frame.name
            copied_frame.write_bytes(b"copied")

            settings = Settings(
                frames_dir=frames_dir,
                detection_metadata_dir=metadata_dir,
            )
            service = DetectionService(settings)

            resolved = service._resolve_metadata_frame_path(copied_frame.resolve())

            self.assertEqual(resolved, canonical_frame.resolve())

    def test_falls_back_to_input_path_when_project_frame_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir = root / "metadata" / "detections"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            copied_frame = root / "isolated_1.0.jpg"
            copied_frame.write_bytes(b"copied")

            settings = Settings(
                frames_dir=frames_dir,
                detection_metadata_dir=metadata_dir,
            )
            service = DetectionService(settings)

            resolved = service._resolve_metadata_frame_path(copied_frame.resolve())

            self.assertEqual(resolved, copied_frame.resolve())

    def test_extract_detection_items_includes_bbox_and_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                frames_dir=root / "frames",
                detection_metadata_dir=root / "metadata" / "detections",
            )
            service = DetectionService(settings)
            result = _FakeResult(
                names={0: "person", 2: "car", 9: "traffic light"},
                cls_values=[2, 9],
                conf_values=[0.91, 0.73],
                bbox_values=[[10, 20, 110, 220], [5, 6, 7, 8]],
            )

            detections = service._extract_detection_items(result)

            self.assertEqual(len(detections), 2)
            self.assertEqual(detections[0].label, "car")
            self.assertEqual(detections[0].class_id, 2)
            self.assertEqual(detections[0].bbox, [10.0, 20.0, 110.0, 220.0])
            self.assertEqual(detections[1].label, "traffic light")
            self.assertEqual(detections[1].class_id, 9)

    def test_detection_video_metadata_backwards_compatible_without_bbox(self) -> None:
        payload = {
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
                        }
                    ],
                }
            ],
        }

        metadata = DetectionVideoMetadata.model_validate(payload)

        self.assertEqual(metadata.frames[0].detections[0].bbox, [])
        self.assertIsNone(metadata.frames[0].detections[0].class_id)

    def test_save_detection_metadata_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                detection_metadata_dir=root / "metadata" / "detections",
            )
            service = DetectionService(settings)

            initial_payload = DetectionVideoMetadata.model_validate(
                {
                    "video_id": "video-1",
                    "video_name": "traffic.mp4",
                    "video_path": "d:/all-seeing vision/rag/project/videos/traffic.mp4",
                    "frames": [],
                }
            )
            updated_payload = DetectionVideoMetadata.model_validate(
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
                                    "class_id": 2,
                                }
                            ],
                        }
                    ],
                }
            )

            service.save_detection_metadata(initial_payload)
            output_path = service.save_detection_metadata(updated_payload)
            loaded = DetectionVideoMetadata.model_validate_json(output_path.read_text(encoding="utf-8"))

            self.assertEqual(len(loaded.frames), 1)
            self.assertEqual(loaded.frames[0].detections[0].bbox, [1.0, 2.0, 3.0, 4.0])

    def test_process_frames_directory_uses_resolved_video_extension_for_video_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            videos_dir = root / "videos"
            detection_dir = root / "metadata" / "detections"
            frames_dir.mkdir(parents=True, exist_ok=True)
            videos_dir.mkdir(parents=True, exist_ok=True)
            detection_dir.mkdir(parents=True, exist_ok=True)

            (videos_dir / "MVI_0001.avi").write_bytes(b"video")
            (frames_dir / "MVI_0001_0.0.jpg").write_bytes(b"frame")

            settings = Settings(
                frames_dir=frames_dir,
                videos_dir=videos_dir,
                detection_metadata_dir=detection_dir,
            )
            service = DetectionService(settings)
            service.detect_frame = lambda frame_path: DetectionFrameMetadata(  # type: ignore[assignment]
                video_name="MVI_0001.mp4",
                frame_path=str(Path(frame_path).resolve()).replace("\\", "/").lower(),
                timestamp=0.0,
                detections=[],
            )

            videos = service.process_frames_directory(frames_dir)

            self.assertEqual(len(videos), 1)
            self.assertEqual(videos[0].video_name, "MVI_0001.avi")
            self.assertEqual(videos[0].frames[0].video_name, "MVI_0001.avi")

    def test_resolves_imported_video_path_from_streetscene_source_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_metadata_dir = root / "metadata" / "datasets"
            dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
            source_dir = root / "datasets" / "StreetScene" / "Test" / "Test001"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_map_path = dataset_metadata_dir / "streetscene_sources.json"
            source_map_path.write_text(
                """
                {
                  "StreetScene_Test_Test001.mp4": {
                    "video_id": "video-1",
                    "video_name": "StreetScene_Test_Test001.mp4",
                    "source_sequence_dir": "SOURCE_DIR_PLACEHOLDER"
                  }
                }
                """.replace("SOURCE_DIR_PLACEHOLDER", str(source_dir).replace("\\", "/")),
                encoding="utf-8",
            )

            settings = Settings(
                dataset_metadata_dir=dataset_metadata_dir,
                detection_metadata_dir=root / "metadata" / "detections",
            )
            service = DetectionService(settings)

            resolved = service._lookup_imported_video_path("StreetScene_Test_Test001.mp4")

            self.assertEqual(resolved, source_dir.resolve())


if __name__ == "__main__":
    unittest.main()
