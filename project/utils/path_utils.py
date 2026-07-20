import hashlib
import re
from pathlib import Path


def normalize_path(path: str | Path) -> str:
    """Normalize a filesystem path for stable indexing and deduplication."""
    return Path(path).expanduser().resolve().as_posix().lower()


def build_asset_id(source_path: Path) -> str:
    normalized = normalize_path(source_path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_path.stem).strip("_") or "video"
    return f"{stem}_{digest}"


def build_frame_id(video_path: str | Path, timestamp_seconds: float) -> str:
    """Build a stable frame identifier from normalized video path and timestamp."""
    normalized_video_path = normalize_path(video_path)
    timestamp_ms = int(round(float(timestamp_seconds) * 1000))
    payload = f"{normalized_video_path}_{timestamp_ms}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
