from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2

from config import Settings


logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi"}


@dataclass
class FrameMetadata:
    frame_path: str
    timestamp: float
    video_name: str

    def to_dict(self) -> dict:
        return asdict(self)


class FrameExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.frames_dir.mkdir(parents=True, exist_ok=True)
        self._traffic_filter = None

    def extract_frames_from_video(
        self,
        video_path: str | Path,
        seconds_per_frame: float = 1.0,
        frame_step: int | None = None,
        clean_output: bool = True,
    ) -> List[dict]:
        resolved_video_path = self._resolve_video_path(video_path)
        self._validate_video_format(resolved_video_path)

        output_dir = self.settings.frames_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if clean_output:
            self._clear_existing_frames_for_video(resolved_video_path.stem)

        capture = cv2.VideoCapture(str(resolved_video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {resolved_video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        effective_frame_step = self._resolve_frame_step(
            fps=fps,
            seconds_per_frame=seconds_per_frame,
            frame_step=frame_step,
        )

        logger.info(
            "Extracting frames from %s with fps=%s and frame_step=%s",
            resolved_video_path,
            round(fps, 3),
            effective_frame_step,
        )

        traffic_filter = self._get_traffic_filter()
        if traffic_filter is not None:
            traffic_filter.reset()
        total_sampled = 0
        total_detected = 0
        filtered_count = 0

        frame_index = 0
        metadata: List[dict] = []

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % effective_frame_step == 0:
                total_sampled += 1

                # Traffic-aware filtering: skip frames with no traffic objects.
                if traffic_filter is not None:
                    should_retain, did_detect, _ = (
                        traffic_filter.should_retain(frame, total_sampled - 1)
                    )
                    if did_detect:
                        total_detected += 1
                    if not should_retain:
                        filtered_count += 1
                        frame_index += 1
                        continue

                # 时间戳保留 2 位小数（与检测/轨迹/OCR 层 S3 精度一致），
                # 避免 0.1s 截断导致相邻采样帧时间戳碰撞、帧文件名互相覆盖丢帧。
                timestamp = round(float(frame_index / fps), 2) if fps else float(frame_index)
                frame_path = output_dir / self._build_frame_filename(
                    video_name=resolved_video_path.stem,
                    timestamp=timestamp,
                )

                if not self._write_frame_image(frame_path, frame):
                    capture.release()
                    raise ValueError(f"Failed to write frame: {frame_path}")

                metadata.append(
                    FrameMetadata(
                        frame_path=str(frame_path),
                        timestamp=timestamp,
                        video_name=resolved_video_path.name,
                    ).to_dict()
                )

            frame_index += 1

        capture.release()

        if traffic_filter is not None:
            retained = len(metadata)
            logger.info(
                "Traffic-aware filter: video=%s total_sampled=%s detected=%s retained=%s filtered=%s",
                resolved_video_path.name,
                total_sampled,
                total_detected,
                retained,
                filtered_count,
            )
        else:
            logger.info(
                "Finished extracting %s frames from %s",
                len(metadata),
                resolved_video_path.name,
            )
        return metadata

    def extract_frames_from_videos(
        self,
        video_paths: Sequence[str | Path],
        seconds_per_frame: float = 1.0,
        clean_output: bool = True,
    ) -> List[dict]:
        all_metadata: List[dict] = []

        for video_path in video_paths:
            all_metadata.extend(
                self.extract_frames_from_video(
                    video_path=video_path,
                    seconds_per_frame=seconds_per_frame,
                    clean_output=clean_output,
                )
            )

        logger.info(
            "Batch extraction completed for %s videos, total frames=%s",
            len(video_paths),
            len(all_metadata),
        )
        return all_metadata

    def extract_frames_from_directory(
        self,
        directory: str | Path,
        seconds_per_frame: float = 1.0,
        clean_output: bool = True,
    ) -> List[dict]:
        video_paths = self.list_supported_videos(directory)
        if not video_paths:
            logger.warning("No supported videos found in %s", directory)
            return []

        return self.extract_frames_from_videos(
            video_paths=video_paths,
            seconds_per_frame=seconds_per_frame,
            clean_output=clean_output,
        )

    def list_supported_videos(self, directory: str | Path) -> List[Path]:
        resolved_directory = Path(directory).resolve()
        if not resolved_directory.exists() or not resolved_directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {resolved_directory}")

        return sorted({path.resolve() for path in self._iter_supported_video_files(resolved_directory)})

    def _iter_supported_video_files(self, directory: Path) -> Iterable[Path]:
        for pattern in ("*.mp4", "*.MP4", "*.avi", "*.AVI"):
            yield from directory.rglob(pattern)

    def _resolve_video_path(self, video_path: str | Path) -> Path:
        path = Path(video_path)
        candidates = []

        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([Path.cwd() / path, self.settings.videos_dir / path])

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Video not found: {video_path}. Place it under {self.settings.videos_dir} or use an absolute path."
        )

    def _validate_video_format(self, video_path: Path) -> None:
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError(
                f"Unsupported video format: {video_path}. Supported formats: {sorted(SUPPORTED_VIDEO_SUFFIXES)}"
            )

    def _build_output_dir(self, video_path: Path) -> Path:
        return self.settings.frames_dir

    def _resolve_frame_step(
        self,
        fps: float,
        seconds_per_frame: float,
        frame_step: int | None,
    ) -> int:
        if frame_step is not None:
            if frame_step < 1:
                raise ValueError("frame_step must be greater than or equal to 1.")
            return frame_step

        if seconds_per_frame <= 0:
            raise ValueError("seconds_per_frame must be greater than 0.")

        if fps <= 0:
            logger.warning("Video fps unavailable, falling back to every frame.")
            return 1

        return max(int(round(fps * seconds_per_frame)), 1)

    def _build_frame_filename(self, video_name: str, timestamp: float) -> str:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_name)
        # 文件名精度与 timestamp 一致（2 位小数），保证同一视频内帧名唯一。
        return f"{safe_name}_{timestamp:.2f}.jpg"

    def _clear_existing_frames_for_video(self, video_name: str) -> None:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_name)
        for existing_frame in self.settings.frames_dir.glob(f"{safe_name}_*.jpg"):
            existing_frame.unlink()

    def _write_frame_image(self, frame_path: Path, frame) -> bool:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("OpenCV failed to encode frame for %s", frame_path)
            return False

        try:
            frame_path.write_bytes(encoded.tobytes())
        except OSError as exc:
            logger.error("Failed to write frame bytes to %s: %s", frame_path, exc)
            return False

        return True

    def _get_traffic_filter(self):
        """Lazily create a TrafficActivityFilter when the feature is enabled."""
        if self._traffic_filter is not None:
            return self._traffic_filter
        if not self.settings.enable_traffic_filter:
            return None
        from services.traffic_activity_filter import TrafficActivityFilter

        self._traffic_filter = TrafficActivityFilter(self.settings)
        logger.info("Traffic-aware video preprocessing layer enabled")
        return self._traffic_filter
