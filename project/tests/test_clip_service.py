"""Tests for the FFmpeg video clip service (mock ffmpeg, no real binary needed).

Verifies:
- ffmpeg missing -> FFmpegUnavailableError
- source video missing -> ClipNotFoundError
- duration > limit -> ClipTooLongError
- successful lossless cut: -ss before -i, -c copy present
- ffmpeg failure -> ClipExecutionError with sanitized stderr
- timeout -> ClipTimeoutError
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import Settings
from services.clip_service import (
    ClipExecutionError,
    ClipInvalidRangeError,
    ClipNotFoundError,
    ClipTimeoutError,
    ClipTooLongError,
    FFmpegUnavailableError,
    VideoClipService,
)


class _FakePopen:
    """Minimal subprocess.Popen stand-in: exit code + optional output file."""

    def __init__(self, cmd, stdout=None, stderr=None, creationflags=0, **kwargs):
        self.cmd = list(cmd)
        self.returncode = 0
        self._stderr = b""
        self._output_path = Path(self.cmd[-1])

    def _make_output(self):
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_bytes(b"FAKE-MP4-BYTES")

    def communicate(self, timeout=None):
        self._make_output()
        return b"", self._stderr

    def kill(self):
        pass


def _settings():
    s = Settings()
    s.clip_max_duration_sec = 60.0
    s.clip_ffmpeg_timeout_sec = 10.0
    return s


def _fake_video(tmp: Path) -> Path:
    video = tmp / "videos" / "监控_MVI_40851.avi"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"FAKE-AVI")
    return video


class VideoClipServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="clip_test_"))
        self.settings = _settings()
        # 白名单适配：测试视频位于临时目录，显式加入允许根目录
        # （Code Review Must Fix #8 安全白名单的可配置项）。
        self.settings.clip_allowed_roots = [self.tmp]
        self.service = VideoClipService(self.settings)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @mock.patch("services.clip_service.shutil.which", return_value=None)
    @mock.patch.object(VideoClipService, "FALLBACK_FFMPEG_PATHS", ())
    def test_ffmpeg_unavailable(self, _mock_which) -> None:
        # Both PATH lookup AND the fallback probe must fail before the error
        # is raised (the fallback would otherwise find a real install).
        video = _fake_video(self.tmp)
        with self.assertRaises(FFmpegUnavailableError):
            self.service.cut_clip(video, 0.0, 1.0)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    def test_video_not_found(self, _mock_which) -> None:
        missing = self.tmp / "no_such.avi"
        with self.assertRaises(ClipNotFoundError):
            self.service.cut_clip(missing, 0.0, 1.0)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    def test_duration_limit(self, _mock_which) -> None:
        video = _fake_video(self.tmp)
        with self.assertRaises(ClipTooLongError):
            self.service.cut_clip(video, 0.0, 61.0)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    def test_invalid_range_rejected(self, _mock_which) -> None:
        """end_ts <= start_ts must fail fast with ClipInvalidRangeError (no
        silent empty-range clipping that would later surface as a 500)."""
        video = _fake_video(self.tmp)
        for start, end in ((5.0, 5.0), (8.0, 3.0), (2.0, -1.0)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ClipInvalidRangeError):
                    self.service.cut_clip(video, start, end)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    @mock.patch("services.clip_service.subprocess.Popen", side_effect=_FakePopen)
    def test_successful_lossless_cut(self, _mock_popen, _mock_which) -> None:
        # clip_reencode=False -> -c copy lossless path.
        self.settings.clip_reencode = False
        video = _fake_video(self.tmp)
        clip = self.service.cut_clip(video, 3.5, 8.5, output_name="red_light_violation_3s")
        self.assertTrue(clip.is_file())
        # -ss must precede -i (fast seek) and -c copy must be present (no re-encode).
        cmd = _mock_popen.call_args.args[0]
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertIn("-c", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertIn("-avoid_negative_ts", cmd)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    @mock.patch("services.clip_service.subprocess.Popen", side_effect=_FakePopen)
    def test_default_reencodes_to_h264(self, _mock_popen, _mock_which) -> None:
        """Default (clip_reencode=True) must emit libx264 + faststart so the
        clip plays in every browser (MPEG-4 Part 2/XVID cannot be decoded by
        <video>)."""
        video = _fake_video(self.tmp)
        self.service.cut_clip(video, 0.0, 2.0)
        cmd = _mock_popen.call_args.args[0]
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
        self.assertIn("-movflags", cmd)
        self.assertEqual(cmd[cmd.index("-movflags") + 1], "+faststart")

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    @mock.patch("services.clip_service.subprocess.Popen")
    def test_ffmpeg_failure_sanitizes_stderr(self, _mock_popen, _mock_which) -> None:
        video = _fake_video(self.tmp)

        def _failing_proc(cmd, stdout=None, stderr=None, creationflags=0, **kwargs):
            proc = _FakePopen(cmd, stdout, stderr, **kwargs)
            proc.returncode = 1
            # Surface the real temporary output path inside stderr so the
            # sanitizer has something concrete to strip.
            proc._stderr = f"ffmpeg error: {cmd[-1]}: No such file".encode()
            return proc

        _mock_popen.side_effect = _failing_proc
        with self.assertRaises(ClipExecutionError) as ctx:
            self.service.cut_clip(video, 0.0, 1.0)
        message = str(ctx.exception)
        # The real temp clip path must never leak into the API-facing message.
        self.assertNotIn(str(self.service.tmp_root), message)
        self.assertIn("ffmpeg failed", message)

    @mock.patch("services.clip_service.shutil.which", return_value="ffmpeg")
    @mock.patch("services.clip_service.subprocess.Popen")
    def test_timeout_kills_process(self, _mock_popen, _mock_which) -> None:
        import subprocess

        video = _fake_video(self.tmp)
        proc = _FakePopen(["ffmpeg", "-i", str(video), "out.mp4"])
        proc.communicate = mock.Mock(
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)
        )
        proc.kill = mock.Mock()
        _mock_popen.return_value = proc
        with self.assertRaises(ClipTimeoutError):
            self.service.cut_clip(video, 0.0, 1.0)
        proc.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
