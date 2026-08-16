from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from config import Settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._embedding_dim = None
        self._load_attempted = False
        self._load_error: str | None = None
        self.settings.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        """Load the CLIP model; a failure marks the service unavailable
        instead of raising, so callers can degrade to other channels."""
        if self._model is not None:
            return
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            self._prepare_download_environment()
            device = self._select_device()
            backend = self.settings.clip_backend.lower()

            if backend == "cnclip":
                self._load_cnclip_model(device)
            else:
                self._load_openclip_model(device)
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.exception("CLIP model loading failed; semantic channel marked unavailable")
            self._model = None
            self._load_error = str(exc)

    def is_available(self) -> bool:
        """Whether the CLIP semantic channel is usable."""
        return self._model is not None

    def _raise_if_unavailable(self) -> None:
        if self._model is None:
            detail = self._load_error or "CLIP model failed to load"
            raise RuntimeError(f"CLIP semantic channel unavailable: {detail}")

    def _select_device(self) -> str:
        import torch

        preferred_device = self.settings.device.lower()
        cuda_available = torch.cuda.is_available()

        if preferred_device == "auto":
            device = "cuda" if cuda_available else "cpu"
        elif preferred_device.startswith("cuda"):
            device = preferred_device if cuda_available else "cpu"
        else:
            device = "cpu"

        if preferred_device.startswith("cuda") and device != "cuda":
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")

        return device

    def _load_openclip_model(self, device: str) -> None:
        import open_clip
        import torch

        logger.info(
            "Loading OpenCLIP model=%s pretrained=%s device=%s",
            "ViT-B-32",
            self.settings.clip_pretrained,
            device,
        )
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained=self.settings.clip_pretrained,
                device=device,
            )
        except Exception as exc:
            logger.exception("Failed to load OpenCLIP weights.")
            raise RuntimeError(
                "Failed to load OpenCLIP weights. Check network access and local proxy settings, then restart the service."
            ) from exc
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = device
        self._embedding_dim = self._infer_embedding_dim()

    def _load_cnclip_model(self, device: str) -> None:
        import torch

        import cn_clip.clip as clip
        from cn_clip.clip import load_from_name

        logger.info(
            "Loading CN-CLIP model=%s device=%s use_modelscope=%s",
            self.settings.cnclip_model_name,
            device,
            self.settings.cnclip_use_modelscope,
        )
        try:
            model, preprocess = load_from_name(
                self.settings.cnclip_model_name,
                device=device,
                download_root=str(self.settings.cnclip_download_root),
                use_modelscope=self.settings.cnclip_use_modelscope,
            )
        except Exception as exc:
            logger.exception("Failed to load CN-CLIP weights.")
            raise RuntimeError(
                "Failed to load CN-CLIP weights. Check network access and modelscope installation, then restart the service."
            ) from exc
        model.eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = clip.tokenize
        self._torch = torch
        self._device = device
        self._embedding_dim = self._infer_embedding_dim()

    def encode_image(self, frame_path: str | Path, timestamp: float | str | None = None) -> dict:
        self.load_model()
        self._raise_if_unavailable()
        resolved_path = self._resolve_image_path(frame_path)

        from PIL import Image

        with Image.open(resolved_path) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            features = self._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        vector = features.cpu().numpy().astype(np.float32)[0]
        result = {
            "frame_path": str(resolved_path),
            "timestamp": self._normalize_timestamp(timestamp, resolved_path),
            "embedding": vector.tolist(),
        }
        logger.info("Encoded frame %s", resolved_path.name)
        return result

    def batch_encode(
        self,
        frame_paths: Sequence[str | Path],
        timestamps: Sequence[float | str] | None = None,
        batch_size: int = 8,
        output_name: str | None = None,
    ) -> List[dict]:
        if not frame_paths:
            raise ValueError("No frame paths provided for batch encoding.")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1.")

        self.load_model()
        self._raise_if_unavailable()

        resolved_paths = [self._resolve_image_path(path) for path in frame_paths]
        normalized_timestamps = self._prepare_timestamps(resolved_paths, timestamps)

        from PIL import Image

        records: List[dict] = []
        for batch_start in range(0, len(resolved_paths), batch_size):
            batch_paths = resolved_paths[batch_start : batch_start + batch_size]
            batch_timestamps = normalized_timestamps[batch_start : batch_start + batch_size]
            images = []

            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(self._preprocess(image.convert("RGB")))

            image_tensor = self._torch.stack(images).to(self._device)

            with self._torch.no_grad():
                features = self._model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            batch_vectors = features.cpu().numpy().astype(np.float32)
            for image_path, timestamp, vector in zip(batch_paths, batch_timestamps, batch_vectors):
                records.append(
                    {
                        "frame_path": str(image_path),
                        "timestamp": timestamp,
                        "embedding": vector.tolist(),
                    }
                )

        if output_name:
            self.save_embeddings(records, output_name)

        logger.info(
            "Batch encoded %s frames using device=%s",
            len(records),
            self._device,
        )
        return records

    def encode_image_paths(self, image_paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
        records = self.batch_encode(frame_paths=image_paths, batch_size=batch_size)
        return np.asarray([record["embedding"] for record in records], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Search query cannot be empty.")

        self.load_model()
        self._raise_if_unavailable()
        tokens = self._tokenizer([text]).to(self._device)

        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)[0]

    def save_embeddings(self, records: Sequence[dict], output_name: str) -> dict:
        if not records:
            raise ValueError("No embedding records to save.")

        output_stem = Path(output_name).stem
        npy_path = self.settings.embeddings_dir / f"{output_stem}.npy"
        json_path = self.settings.embeddings_dir / f"{output_stem}.json"

        matrix = np.asarray([record["embedding"] for record in records], dtype=np.float32)
        json_records = [
            {
                "frame_path": record["frame_path"],
                "timestamp": record["timestamp"],
            }
            for record in records
        ]

        np.save(npy_path, matrix)
        json_path.write_text(
            json.dumps(json_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %s embeddings to %s", len(records), npy_path)
        return {
            "npy_path": str(npy_path),
            "json_path": str(json_path),
            "count": len(records),
            "embedding_dim": int(matrix.shape[1]),
        }

    def get_embedding_dim(self) -> int:
        self.load_model()
        self._raise_if_unavailable()
        return int(self._embedding_dim or 0)

    def save_segment_embeddings(
        self,
        segments: Sequence[dict],
        vectors: np.ndarray,
        model_name: str,
    ) -> List[dict]:
        """S4: 每 segment 存一个 .npy（embeddings/{segment_id}.npy），
        返回 embeddings meta 列表，由调用方聚合进 metadata/segments/{video_id}.json
        （数据规范 v1.1 §3.9，段向量 meta 不另设文件）。

        参数:
          segments: 段记录列表，每项必须含 segment_id（可选 text_source / created_at）。
          vectors:  形状 (N, dim) 的向量矩阵，N == len(segments)。
          model_name: 编码模型名（如 "CN-CLIP" / "OpenCLIP"），写入 meta 供溯源。
        """
        if len(segments) != len(vectors):
            raise ValueError("Segments count must match vectors count.")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D numpy array.")

        self.settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
        metas: List[dict] = []
        for segment, vector in zip(segments, vectors):
            segment_id = segment["segment_id"]
            npy_path = self.settings.embeddings_dir / f"{segment_id}.npy"
            np.save(npy_path, np.asarray(vector, dtype=np.float32))
            metas.append(
                {
                    "segment_id": segment_id,
                    "model": model_name,
                    "dimension": int(np.asarray(vector).shape[0]),
                    "path": f"embeddings/{segment_id}.npy",
                    "text_source": segment.get("text_source", "事件描述模板"),
                    "created_at": segment.get("created_at", ""),
                }
            )
        logger.info("Saved %s segment embeddings (model=%s)", len(metas), model_name)
        return metas

    def _infer_embedding_dim(self) -> int:
        if hasattr(self._model, "visual") and hasattr(self._model.visual, "output_dim"):
            return int(self._model.visual.output_dim)

        probe = self.encode_text("dimension probe")
        return int(probe.shape[0])

    def _prepare_timestamps(
        self,
        frame_paths: Sequence[Path],
        timestamps: Sequence[float | str] | None,
    ) -> List[str]:
        if timestamps is not None and len(timestamps) != len(frame_paths):
            raise ValueError("timestamps length must match frame_paths length.")

        if timestamps is None:
            return [self._normalize_timestamp(None, frame_path) for frame_path in frame_paths]

        return [
            self._normalize_timestamp(timestamp, frame_path)
            for frame_path, timestamp in zip(frame_paths, timestamps)
        ]

    def _normalize_timestamp(self, timestamp: float | str | None, frame_path: Path) -> str:
        if timestamp is not None:
            # S3: 统一两位小数，保证跨层 timestamp 一致（数据规范 v1.1 §0）。
            return str(round(float(timestamp), 2))

        stem_parts = frame_path.stem.rsplit("_", 1)
        if len(stem_parts) == 2:
            try:
                # S3: 统一两位小数
                return str(round(float(stem_parts[1]), 2))
            except ValueError:
                # 帧名末段不是时间戳（如自定义命名），兜底为 0.0
                logger.warning("Frame name has no parseable timestamp, defaulting to 0.0: %s", frame_path)
                return "0.0"

        return "0.0"

    def _resolve_image_path(self, frame_path: str | Path) -> Path:
        path = Path(frame_path)
        candidates: Iterable[Path]

        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [Path.cwd() / path, self.settings.frames_dir / path]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(f"Frame image not found: {frame_path}")

    def _prepare_download_environment(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        proxy_keys = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
        proxy_markers = ("127.0.0.1:7890", "localhost:7890")

        for key in proxy_keys:
            value = os.environ.get(key)
            if value and any(marker in value for marker in proxy_markers):
                logger.warning("Clearing unavailable local proxy from %s=%s", key, value)
                os.environ.pop(key, None)
