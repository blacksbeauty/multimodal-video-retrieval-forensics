from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from config import Settings
from services.frame_extractor import FrameExtractor
from utils.path_utils import build_asset_id


logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.frame_extractor = FrameExtractor(settings)

    def _resolve_video_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        candidates = []

        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                [
                    Path.cwd() / path,
                    self.settings.videos_dir / path,
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Video not found: {raw_path}. Place it under {self.settings.videos_dir} or use an absolute path."
        )

    def ingest_video(
        self,
        video_path: str,
        frame_interval: int,
        clip_service,
        index_service,
        save_index: bool = True,
    ) -> Dict:
        """单视频摄取（抽帧+CLIP+FAISS 增量索引）。

        批量调用方（ingest_directory/数据集导入）传 save_index=False，
        在循环外统一 save_index 一次，避免 N 次全量落盘（Code Review P1-1）。
        """
        if frame_interval < 1:
            raise ValueError("frame_interval must be greater than or equal to 1.")

        import cv2

        resolved_video_path = self._resolve_video_path(video_path)
        video_id = build_asset_id(resolved_video_path)

        capture = cv2.VideoCapture(str(resolved_video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {resolved_video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        extracted_metadata = self.frame_extractor.extract_frames_from_video(
            video_path=resolved_video_path,
            frame_step=frame_interval,
            clean_output=True,
        )

        if not extracted_metadata:
            raise ValueError(f"No frames extracted from video: {resolved_video_path}")

        extracted_paths: List[Path] = [Path(item["frame_path"]) for item in extracted_metadata]
        frame_metadata: List[Dict] = []

        for item in extracted_metadata:
            timestamp = float(item["timestamp"])
            frame_metadata.append(
                {
                    "video_id": video_id,
                    "video_name": item["video_name"],
                    "video_path": str(resolved_video_path),
                    "frame_path": item["frame_path"],
                    "frame_index": int(round(timestamp * fps)) if fps else 0,
                    "timestamp_seconds": timestamp,
                }
            )

        embeddings = clip_service.encode_image_paths(extracted_paths)
        index_service.upsert_video_records(video_id, frame_metadata, embeddings, save=save_index)

        logger.info(
            "Completed video ingest path=%s extracted_frames=%s",
            resolved_video_path,
            len(extracted_paths),
        )
        return {
            "video_name": resolved_video_path.name,
            "video_path": str(resolved_video_path),
            "total_frames": total_frames,
            "extracted_frames": len(extracted_paths),
            "indexed_frames": len(frame_metadata),
        }

    def ingest_directory(self, directory: str | None, frame_interval: int, clip_service, index_service) -> Dict:
        resolved_directory = Path(directory).resolve() if directory else self.settings.videos_dir.resolve()
        video_paths = self.frame_extractor.list_supported_videos(resolved_directory)

        if not video_paths:
            raise FileNotFoundError(f"No supported videos found in directory: {resolved_directory}")

        results: List[Dict] = []
        errors: List[str] = []

        for video_path in video_paths:
            try:
                results.append(
                    self.ingest_video(
                        video_path=str(video_path),
                        frame_interval=frame_interval,
                        clip_service=clip_service,
                        index_service=index_service,
                        # 批量：循环内不落盘，结束统一 save（P1-1）
                        save_index=False,
                    )
                )
            except Exception as exc:
                logger.exception("Failed to ingest video during batch processing: %s", video_path)
                errors.append(f"{video_path.name}: {exc}")

        if results:
            index_service.faiss_service.save_index()

        if not results and errors:
            raise ValueError(
                "All videos failed during batch ingest. Check OpenCV codec support and frame write permissions."
            )

        return {
            "total_videos": len(video_paths),
            "succeeded_videos": len(results),
            "failed_videos": len(errors),
            "results": results,
            "errors": errors,
        }
