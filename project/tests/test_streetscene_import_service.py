import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings
from services.streetscene_import_service import StreetSceneImportService


class _FakeClipService:
    def encode_image_paths(self, image_paths, batch_size=8):
        return np.ones((len(image_paths), 4), dtype=np.float32)


class _FakeIndexService:
    def __init__(self):
        self.calls = []

    def upsert_video_records(self, video_id, frame_metadata, embeddings):
        self.calls.append(
            {
                "video_id": video_id,
                "frame_metadata": frame_metadata,
                "embeddings_shape": tuple(embeddings.shape),
            }
        )


class StreetSceneImportServiceTests(unittest.TestCase):
    def test_import_sequence_maps_frames_into_foresea_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "StreetScene"
            sequence_dir = dataset_root / "Test" / "Test001"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = root / "frames"
            dataset_metadata_dir = root / "metadata" / "datasets"

            for name in ("00001.jpg", "00002.jpg", "00003.jpg"):
                Image.new("RGB", (32, 32), color="white").save(sequence_dir / name)

            settings = Settings(
                frames_dir=frames_dir,
                dataset_metadata_dir=dataset_metadata_dir,
                streetscene_frame_rate=15.0,
            )
            service = StreetSceneImportService(settings)
            clip_service = _FakeClipService()
            index_service = _FakeIndexService()

            result = service.ingest_directory(
                dataset_root=dataset_root,
                split="Test",
                sequence_names=["Test001"],
                max_sequences=None,
                frame_step=1,
                clip_service=clip_service,
                index_service=index_service,
            )

            self.assertEqual(result["succeeded_sequences"], 1)
            self.assertEqual(len(index_service.calls), 1)
            metadata = index_service.calls[0]["frame_metadata"]
            self.assertEqual(len(metadata), 3)
            self.assertTrue(metadata[0]["frame_path"].endswith("streetscene_test_test001_0.000.jpg"))
            self.assertTrue(metadata[1]["frame_path"].endswith("streetscene_test_test001_0.067.jpg"))
            self.assertEqual(index_service.calls[0]["embeddings_shape"], (3, 4))

            source_map_path = dataset_metadata_dir / "streetscene_sources.json"
            self.assertTrue(source_map_path.exists())
            source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
            self.assertIn("StreetScene_Test_Test001.mp4", source_map)


if __name__ == "__main__":
    unittest.main()
