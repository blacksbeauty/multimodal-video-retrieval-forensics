from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2

from config import Settings
from core.schemas import OCRFrameMetadata, OCRTextResult, OCRVideoMetadata
from services.ocr_metadata_store import OCRMetadataStore
from utils.frame_utils import read_image_cv


logger = logging.getLogger(__name__)


class OCRService:
    """Run PaddleOCR on extracted video frames and persist OCR metadata per video."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the OCR service with configuration, output directories, and load guards."""
        self.settings = settings
        self.ocr = None
        self.metadata_store = OCRMetadataStore(settings)
        self.model_load_attempted = False
        self.model_available = False
        self.model_load_error: str | None = None

        self.settings.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.settings.ocr_metadata_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> bool:
        """Load PaddleOCR once in CPU-only mode and fuse repeated failures."""
        if self.model_available and self.ocr is not None:
            return True

        if self.model_load_attempted and not self.model_available:
            return False

        self.model_load_attempted = True

        try:
            import torch  # Load torch first to reduce Windows DLL resolution conflicts.

            logger.info("Torch preloaded before PaddleOCR import: version=%s", torch.__version__)
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self.model_load_error = (
                "PaddleOCR dependencies are not installed correctly. Please install paddleocr, paddlepaddle, and torch."
            )
            logger.exception(self.model_load_error)
            return False
        except OSError as exc:
            self.model_load_error = (
                "PaddleOCR import failed due to a Windows DLL dependency error. "
                "This often means torch or one of its native dependencies is not usable in the current environment."
            )
            logger.exception(self.model_load_error)
            return False

        try:
            logger.info(
                "Loading PaddleOCR in CPU-only mode. paddle_cuda_available=%s",
                paddle.device.is_compiled_with_cuda(),
            )
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=False,
                show_log=False,
            )
        except Exception as exc:
            self.model_load_error = f"Failed to initialize PaddleOCR in CPU-only mode: {exc}"
            logger.exception("Failed to initialize PaddleOCR.")
            self.ocr = None
            self.model_available = False
            return False

        self.model_available = True
        self.model_load_error = None
        logger.info("PaddleOCR initialized successfully in CPU-only mode.")
        return True

    def extract_text_from_frame(self, frame_path: str | Path) -> List[OCRTextResult]:
        """Run OCR on a single frame image and return normalized OCR text regions."""
        if not self.load_model():
            return []

        resolved_frame_path = Path(frame_path).resolve()
        # 中文路径兼容读取（Code Review 修复：cv2.imread 无法打开含中文路径的帧）
        image = read_image_cv(resolved_frame_path)
        if image is None:
            raise ValueError(f"Failed to read frame image: {resolved_frame_path}")

        raw_result = self.ocr.ocr(image, cls=True)
        if not raw_result or not raw_result[0]:
            logger.info("No OCR text detected for frame %s", resolved_frame_path.name)
            return []

        results: List[OCRTextResult] = []
        for item in raw_result[0]:
            if not item or len(item) < 2:
                continue

            bbox_raw, text_info = item
            # 防御（Code Review Must Fix #3）：PaddleOCR 可能返回 [None, score]
            # （未识别到有效文本），str(None)="None" 会被误当真实文本入库。
            if not text_info or len(text_info) < 1 or text_info[0] is None:
                continue
            text = str(text_info[0]).strip()
            score = float(text_info[1]) if len(text_info) > 1 else 0.0
            # 防御：bbox 结构异常（点数不足 / 坐标缺失）时跳过该条，避免 IndexError。
            try:
                bbox = [[int(round(point[0])), int(round(point[1]))] for point in bbox_raw]
            except (IndexError, TypeError, ValueError):
                logger.warning(
                    "Skipping OCR region with malformed bbox on frame %s", resolved_frame_path.name
                )
                continue

            if not text:
                continue

            results.append(
                OCRTextResult(
                    text=text,
                    score=score,
                    bbox=bbox,
                )
            )

        logger.info(
            "OCR extracted %s text regions from frame %s",
            len(results),
            resolved_frame_path.name,
        )
        return results

    def process_frame(self, frame_path: str | Path) -> OCRFrameMetadata | None:
        """Process one frame safely and return structured OCR metadata."""
        resolved_frame_path = Path(frame_path).resolve()

        if not self.model_available and self.model_load_attempted:
            logger.debug("Skipping OCR for frame %s because OCR model is unavailable.", resolved_frame_path)
            return None

        try:
            ocr_results = self.extract_text_from_frame(resolved_frame_path)
        except Exception:
            logger.exception("OCR failed for frame %s", resolved_frame_path)
            return None

        if not self.model_available:
            return None

        timestamp_sec = self._extract_timestamp_from_frame_name(resolved_frame_path)
        # S3: 统一两位小数，保证与检测/轨迹/事件层 timestamp 可跨层对齐（数据规范 v1.1 §0）。
        timestamp_sec = round(timestamp_sec, 2)
        return OCRFrameMetadata(
            frame_path=str(resolved_frame_path),
            timestamp_sec=timestamp_sec,
            display_time=self._format_display_time(timestamp_sec),
            ocr_results=ocr_results,
        )

    def process_directory(self, directory: str | Path | None = None) -> List[OCRVideoMetadata]:
        """Process every frame in a directory, group by video name, and save OCR metadata per video."""
        target_directory = Path(directory).resolve() if directory else self.settings.frames_dir.resolve()
        if not target_directory.exists() or not target_directory.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {target_directory}")

        frame_paths = sorted(target_directory.glob("*.jpg"))
        if not frame_paths:
            logger.warning("No frame images found in %s", target_directory)
            return []

        grouped_frames: Dict[str, List[OCRFrameMetadata]] = {}

        logger.info("Starting OCR batch processing for %s frames", len(frame_paths))
        for frame_path in frame_paths:
            frame_metadata = self.process_frame(frame_path)
            if frame_metadata is None:
                continue

            video_name = self._extract_video_name_from_frame(frame_path)
            grouped_frames.setdefault(video_name, []).append(frame_metadata)

        videos: List[OCRVideoMetadata] = []
        for video_name, frames in grouped_frames.items():
            ordered_frames = sorted(frames, key=lambda item: item.timestamp_sec)
            video_metadata = OCRVideoMetadata(video_name=video_name, frames=ordered_frames)
            self.save_video_metadata(video_metadata)
            videos.append(video_metadata)

        logger.info(
            "Completed OCR batch processing. videos=%s processed_frames=%s",
            len(videos),
            sum(len(video.frames) for video in videos),
        )
        return videos

    def save_video_metadata(self, video_metadata: OCRVideoMetadata) -> Path:
        """Persist OCR metadata for a single video as UTF-8 JSON under metadata/ocr."""
        return self.metadata_store.save_metadata(video_metadata, overwrite=True)

    def _extract_timestamp_from_frame_name(self, frame_path: Path) -> float:
        stem_parts = frame_path.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            return 0.0

        try:
            return float(stem_parts[1])
        except ValueError:
            return 0.0

    def _extract_video_name_from_frame(self, frame_path: Path) -> str:
        stem_parts = frame_path.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            return frame_path.stem
        return stem_parts[0]

    def _format_display_time(self, timestamp_sec: float) -> str:
        total_seconds = int(timestamp_sec)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
