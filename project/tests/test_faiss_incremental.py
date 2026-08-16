"""Regression tests for the incremental FAISS index (IndexIDMap) rewrite.

Covers:
- incremental add of a new video (no full rebuild)
- overwriting an existing video (remove old ids + re-add)
- save -> load round-trip (including legacy list metadata conversion)
- search resolves external ids back to frame metadata
- remove_video cleanup
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Settings
from services.faiss_index_service import FaissIndexService


class FaissIncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.mkdtemp(prefix="faiss_incr_")
        self.settings = Settings()
        self.settings.index_dir = Path(tmp) / "index"
        self.settings.faiss_index_path = Path(tmp) / "index" / "video_frames.index"
        self.settings.metadata_path = Path(tmp) / "index" / "frame_metadata.json"
        self.settings.embeddings_dir = Path(tmp) / "embeddings"

    def _make(self, video_id: str, n: int, dim: int = 8):
        embeddings = np.random.rand(n, dim).astype(np.float32)
        metadata = [
            {
                "video_id": video_id,
                "video_name": f"{video_id}.mp4",
                "video_path": f"videos/{video_id}.mp4",
                "frame_path": f"frames/{video_id}_{i}.jpg",
                "timestamp_seconds": float(i),
                "frame_id": f"{video_id}:{i}",
            }
            for i in range(n)
        ]
        return embeddings, metadata

    def test_incremental_add_overwrite_remove(self) -> None:
        svc = FaissIndexService(self.settings)

        # Add a first video.
        emb1, meta1 = self._make("v1", 3)
        svc.add_video("v1", emb1, meta1)
        self.assertEqual(svc.index.ntotal, 3)
        self.assertEqual(svc.mapping["v1"]["count"], 3)

        # Incrementally append a second video — total grows, nothing else touched.
        emb2, meta2 = self._make("v2", 2)
        svc.add_video("v2", emb2, meta2)
        self.assertEqual(svc.index.ntotal, 5)
        self.assertIn("v2", svc.mapping)

        # Overwrite v1 with 4 vectors: old 3 ids removed, 4 new added → 2+4=6.
        emb1b, meta1b = self._make("v1", 4)
        svc.add_video("v1", emb1b, meta1b)
        self.assertEqual(svc.index.ntotal, 6)
        self.assertEqual(len(svc.metadata), 6)
        self.assertEqual(svc.mapping["v1"]["count"], 4)

        # Persist and reload into a fresh service.
        svc.save_index()
        svc2 = FaissIndexService(self.settings)
        svc2.load_index()
        self.assertEqual(svc2.index.ntotal, 6)
        self.assertEqual(len(svc2.metadata), 6)
        self.assertIn("v1", svc2.mapping)

        # Search resolves external ids to the metadata of the newest v1 vectors.
        results = svc2.search(emb1b[0], top_k=6)
        self.assertEqual(len(results), 6)
        found_v1_frames = {r["frame_path"] for r in results if r.get("video_id") == "v1"}
        self.assertTrue(found_v1_frames)

        # Remove v2 entirely.
        svc2.remove_video("v2")
        self.assertEqual(svc2.index.ntotal, 4)
        self.assertNotIn("v2", svc2.mapping)
        self.assertNotIn("v2", {item.get("video_id") for item in svc2.metadata.values()})

    def test_full_build_creates_mapping_and_saves(self) -> None:
        svc = FaissIndexService(self.settings)
        emb, meta = self._make("v1", 5)
        svc.build_index(emb, meta)
        self.assertEqual(svc.index.ntotal, 5)
        self.assertEqual(svc.mapping["v1"]["count"], 5)
        self.assertEqual(len(svc.mapping["v1"]["ids"]), 5)
        svc.save_index()

        # A fresh service can load the built index and search.
        svc2 = FaissIndexService(self.settings)
        svc2.load_index()
        self.assertEqual(svc2.index.ntotal, 5)
        self.assertEqual(len(svc2.search(emb[0], top_k=5)), 5)

    def test_load_converts_legacy_list_metadata(self) -> None:
        svc = FaissIndexService(self.settings)
        emb, meta = self._make("v1", 4)
        svc.build_index(emb, meta)
        svc.save_index()

        # Simulate the legacy metadata payload (a plain list).
        self.settings.metadata_path.write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )
        svc2 = FaissIndexService(self.settings)
        svc2.load_index()
        self.assertEqual(len(svc2.metadata), 4)
        self.assertEqual(svc2.index.ntotal, 4)


if __name__ == "__main__":
    unittest.main()
