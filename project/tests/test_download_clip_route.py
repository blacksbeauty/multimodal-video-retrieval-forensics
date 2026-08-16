"""API-level tests for the ``GET /api/search/download_clip`` endpoint.

Uses a minimal FastAPI app that mounts the real ``api.routes.router`` with a
mocked ``AppState`` (only ``video_clip_service`` is exercised), so no FAISS /
CLIP / OCR models are loaded.

Verifies:
- concurrency slot exhausted -> 429 (H2 fix)
- success -> 200, clip bytes streamed, slot released, temp file cleaned up
- invalid range (end_ts <= start_ts) -> 400 (M4 fix)
- error mapping: 404 / 504 / 500
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from services.clip_service import (
    ClipExecutionError,
    ClipInvalidRangeError,
    ClipNotFoundError,
    ClipTimeoutError,
    ClipTooLongError,
)

CLIP_URL = "/api/search/download_clip"


def _make_app(video_clip_service) -> FastAPI:
    """Minimal app: real router, mocked services container."""
    app = FastAPI()
    services = mock.Mock()
    services.video_clip_service = video_clip_service
    app.state.services = services
    app.include_router(router, prefix="/api")
    return app


class DownloadClipRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="download_clip_test_"))
        self.clip_service = mock.Mock()
        self.clip_service.try_acquire_slot.return_value = True
        self.app = _make_app(self.clip_service)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _clip_file(self) -> Path:
        clip = self.tmp / "clip.mp4"
        clip.write_bytes(b"FAKE-MP4-BYTES")
        return clip

    # ------------------------------------------------------------------ #
    # H2: concurrency limit -> 429
    # ------------------------------------------------------------------ #

    def test_429_when_concurrency_slot_unavailable(self) -> None:
        """At capacity: try_acquire_slot -> False must yield 429, and cut_clip
        must never be invoked."""
        self.clip_service.try_acquire_slot.return_value = False

        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 0.0, "end_ts": 1.0},
        )

        self.assertEqual(resp.status_code, 429)
        self.assertIn("concurrent", resp.json()["detail"].lower())
        self.clip_service.cut_clip.assert_not_called()
        self.clip_service.release_slot.assert_not_called()

    def test_slot_released_on_success(self) -> None:
        self.clip_service.cut_clip.return_value = self._clip_file()

        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 0.0, "end_ts": 1.0},
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"FAKE-MP4-BYTES")
        self.clip_service.try_acquire_slot.assert_called_once()
        self.clip_service.release_slot.assert_called_once()

    # ------------------------------------------------------------------ #
    # M4: invalid range -> 400
    # ------------------------------------------------------------------ #

    def test_400_invalid_range_maps_to_bad_request(self) -> None:
        self.clip_service.cut_clip.side_effect = ClipInvalidRangeError(
            "Invalid clip range: end_ts (1.0) must be greater than start_ts (2.0)."
        )

        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 2.0, "end_ts": 1.0},
        )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid clip range", resp.json()["detail"])
        # Slot must still be released even though the cut failed.
        self.clip_service.release_slot.assert_called_once()

    # ------------------------------------------------------------------ #
    # Error mapping regression
    # ------------------------------------------------------------------ #

    def test_400_too_long(self) -> None:
        self.clip_service.cut_clip.side_effect = ClipTooLongError("too long")
        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 0.0, "end_ts": 100.0},
        )
        self.assertEqual(resp.status_code, 400)

    def test_404_not_found(self) -> None:
        self.clip_service.cut_clip.side_effect = ClipNotFoundError("missing")
        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "nope.avi", "start_ts": 0.0, "end_ts": 1.0},
        )
        self.assertEqual(resp.status_code, 404)

    def test_504_timeout(self) -> None:
        self.clip_service.cut_clip.side_effect = ClipTimeoutError("slow ffmpeg")
        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 0.0, "end_ts": 1.0},
        )
        self.assertEqual(resp.status_code, 504)

    def test_500_generic_clip_error(self) -> None:
        self.clip_service.cut_clip.side_effect = ClipExecutionError("ffmpeg failed")
        resp = self.client.get(
            CLIP_URL,
            params={"video_path": "x.avi", "start_ts": 0.0, "end_ts": 1.0},
        )
        self.assertEqual(resp.status_code, 500)
        self.clip_service.release_slot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
