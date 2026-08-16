from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from config import Settings


logger = logging.getLogger(__name__)


class ClipError(Exception):
    """Base error for video clipping failures."""


class FFmpegUnavailableError(ClipError):
    """Raised when the ffmpeg binary is not installed."""


class ClipNotFoundError(ClipError):
    """Raised when the source video file does not exist."""


class ClipTooLongError(ClipError):
    """Raised when the requested clip exceeds the maximum duration."""


class ClipInvalidRangeError(ClipError):
    """Raised when ``end_ts <= start_ts`` — the requested range is empty."""


class ClipTimeoutError(ClipError):
    """Raised when ffmpeg does not finish within the allowed timeout."""


class ClipExecutionError(ClipError):
    """Raised when ffmpeg exits with a non-zero return code."""


class VideoClipService:
    """Lossless video clipping via FFmpeg, tuned for edge deployment.

    * ``-ss`` is placed *before* ``-i`` for fast seeking.
    * ``-c copy`` performs a stream copy (no re-encode).
    * Clips are written to an in-memory filesystem (``/dev/shm``) when
      available and to the OS temp dir otherwise, then removed asynchronously
      after the HTTP response is sent.
    """

    # Default limits (override via Settings).
    DEFAULT_MAX_DURATION_SEC = 60.0
    DEFAULT_FFMPEG_TIMEOUT_SEC = 10.0
    DEFAULT_MAX_CONCURRENT = 2

    # When ffmpeg is not on PATH (e.g. the server process predates an
    # installation), probe common install locations before giving up.
    FALLBACK_FFMPEG_PATHS: tuple[str, ...] = (
        r"D:\software\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ffmpeg: Optional[str] = None
        self._tmp_root: Optional[Path] = None
        max_concurrent = max(
            1,
            int(getattr(settings, "clip_max_concurrent", self.DEFAULT_MAX_CONCURRENT)),
        )
        # Guards how many ffmpeg cuts may run at once (each spawns a subprocess
        # and can saturate CPU). Thread-safe because the endpoint runs in
        # FastAPI's thread pool (sync def), not on the event loop.
        self._slot_sem = threading.BoundedSemaphore(max_concurrent)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def ffmpeg_path(self) -> str:
        """Path to the ffmpeg binary; raises if not installed."""
        if self._ffmpeg is None:
            self._ffmpeg = shutil.which("ffmpeg")
            if not self._ffmpeg:
                # PATH may be stale (server predates installation); probe
                # common locations so a code reload picks ffmpeg up without
                # a full process restart.
                self._ffmpeg = next(
                    (
                        candidate
                        for candidate in self.FALLBACK_FFMPEG_PATHS
                        if Path(candidate).is_file()
                    ),
                    None,
                )
        if not self._ffmpeg:
            raise FFmpegUnavailableError(
                "ffmpeg is not installed or not on PATH. "
                "Please install FFmpeg (e.g. `apt install ffmpeg`, "
                "`brew install ffmpeg`, or download from https://ffmpeg.org)."
            )
        return self._ffmpeg

    @property
    def tmp_root(self) -> Path:
        """Temporary clip directory — /dev/shm when available, else OS temp."""
        if self._tmp_root is None:
            self._tmp_root = self._resolve_tmp_root()
        return self._tmp_root

    def try_acquire_slot(self) -> bool:
        """Try to reserve one of the limited ffmpeg concurrency slots.

        Returns ``True`` when a slot was acquired — the caller MUST pair this
        with :meth:`release_slot` (prefer ``try/finally``). Returns ``False``
        when at capacity so the API layer can answer ``429``.
        """
        return self._slot_sem.acquire(blocking=False)

    def release_slot(self) -> None:
        """Release a previously acquired ffmpeg concurrency slot."""
        self._slot_sem.release()

    def cut_clip(
        self,
        video_path: str | Path,
        start_ts: float,
        end_ts: float,
        output_name: str | None = None,
    ) -> Path:
        """Losslessly cut ``[start_ts, end_ts]`` from ``video_path``.

        Returns the path of the generated MP4 clip. The caller is responsible
        for scheduling :meth:`cleanup` on that path once the response is sent.
        """
        ffmpeg = self.ffmpeg_path  # raises FFmpegUnavailableError early

        source = Path(video_path)
        if not source.is_file():
            raise ClipNotFoundError(f"Video file not found: {source}")
        if not self._is_allowed_source(source):
            # Code Review Must Fix #8：只允许剪辑受管视频目录内的源文件，
            # 防止 download_clip 端点被用于读取服务器任意媒体文件（返回 404 不泄露信息）。
            logger.warning("Blocked clip request for non-whitelisted video path: %s", source)
            raise ClipNotFoundError(f"Video file not found: {source}")

        start = max(float(start_ts), 0.0)
        end = max(float(end_ts), 0.0)
        if end <= start:
            raise ClipInvalidRangeError(
                f"Invalid clip range: end_ts ({end_ts}) must be greater "
                f"than start_ts ({start_ts})."
            )
        duration = end - start
        max_duration = float(
            getattr(self.settings, "clip_max_duration_sec", self.DEFAULT_MAX_DURATION_SEC)
        )
        if duration > max_duration:
            raise ClipTooLongError(
                f"Requested clip duration {duration:.2f}s exceeds the "
                f"limit of {max_duration:.0f}s."
            )

        self.tmp_root.mkdir(parents=True, exist_ok=True)
        safe_stem = self._safe_stem(output_name or source.stem)
        output_path = self.tmp_root / f"{safe_stem}_{int(time.time() * 1000)}_{os.getpid()}.mp4"

        timeout = float(
            getattr(self.settings, "clip_ffmpeg_timeout_sec", self.DEFAULT_FFMPEG_TIMEOUT_SEC)
        )
        # -ss BEFORE -i for fast seek.
        # Browser <video> cannot decode MPEG-4 Part 2 (XVID/mp4v) that -c copy
        # would preserve from AVI sources; default to H.264 re-encode so the
        # clip always plays. Set clip_reencode=False for lossless copy (only
        # when the source codec is browser-friendly, e.g. H.264/HEVC/VP9).
        # -movflags +faststart moves the moov atom to the file head so the
        # browser can start streaming from a Range/206 response.
        reencode = bool(getattr(self.settings, "clip_reencode", True))
        codec_args = (
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac"]
            if reencode
            else ["-c", "copy"]
        )
        command = [
            ffmpeg,
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            *codec_args,
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info("Cutting clip video=%s start=%.2f duration=%.2f", source.name, start, duration)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:  # pragma: no cover - best effort reap
                pass
            logger.error("FFmpeg timed out after %.1fs for %s", timeout, source.name)
            raise ClipTimeoutError(
                f"FFmpeg did not finish within {timeout:.0f}s; process terminated."
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            logger.error(
                "FFmpeg failed code=%s video=%s stderr=%s",
                proc.returncode,
                source.name,
                stderr_text[-400:],
            )
            raise ClipExecutionError(self._sanitize_stderr(stderr_text, source, output_path))

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ClipExecutionError("FFmpeg produced an empty output file.")

        logger.info("Clip created %s bytes=%s", output_path.name, output_path.stat().st_size)
        return output_path

    def cleanup(self, path: str | Path) -> None:
        """Best-effort removal of a temporary clip (safe to call once)."""
        try:
            target = Path(path)
            if target.is_file():
                target.unlink(missing_ok=True)
                logger.debug("Removed temporary clip %s", target.name)
        except OSError:  # pragma: no cover - never raise from cleanup
            logger.warning("Failed to remove temporary clip %s", path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_allowed_source(self, source: Path) -> bool:
        """校验视频源是否位于受管媒体目录内（防任意文件读取）。

        白名单根目录：videos_dir + 两个数据集目录（streetscene/accident），
        以及 settings.clip_allowed_roots 中显式挂载的额外目录（测试/第三方视频）。
        不存在的根目录直接跳过；Windows 下 Path.is_relative_to 大小写不敏感，
        因此 metadata 中的小写化路径也能正确匹配。
        """
        resolved = source.resolve()
        roots = (
            self.settings.videos_dir,
            self.settings.streetscene_dataset_dir,
            self.settings.accident_dataset_dir,
            *tuple(getattr(self.settings, "clip_allowed_roots", ()) or ()),
        )
        for root in roots:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                continue
            try:
                if resolved.is_relative_to(root_path.resolve()):
                    return True
            except ValueError:  # 不同盘符等无法比较的情况
                continue
        return False

    def _resolve_tmp_root(self) -> Path:
        # Edge devices (Linux): prefer in-memory /dev/shm to avoid flash wear.
        if os.name == "posix" and Path("/dev/shm").is_dir():
            root = Path("/dev/shm") / "clips"
        else:
            root = Path(tempfile.gettempdir()) / "clips"
        root.mkdir(parents=True, exist_ok=True)
        logger.info("Video clip temp root: %s", root)
        return root

    def _safe_stem(self, name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "clip"
        return cleaned[:80]

    def _sanitize_stderr(self, stderr_text: str, source: Path, output_path: Path) -> str:
        """Strip absolute paths from ffmpeg stderr before surfacing it to the API."""
        for sensitive in (str(output_path), str(source), str(self.tmp_root)):
            stderr_text = stderr_text.replace(sensitive, "<path>")
        lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        # Keep only the most relevant tail (usually the actual error line).
        tail = lines[-6:]
        return "ffmpeg failed. " + " | ".join(tail[-2:]) if tail else "ffmpeg failed."
