#!/usr/bin/env python
"""Migrate frame embeddings from OpenCLIP to CN-CLIP.

Re-encodes all video frames using the CN-CLIP ViT-B-16 model and rebuilds
the FAISS index. Old embeddings are archived for rollback.

Usage:
    cd project/
    python tests/migrate_to_cnclip.py [--dry-run] [--backup-suffix openclip_backup]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from config import get_settings
from services.embedding_service import EmbeddingService
from services.index_service import IndexService


def resolve_frame_path(old_frame_path: str, frames_dir: Path) -> Path | None:
    """Resolve an old frame path to the current frames directory by filename."""
    frame_name = Path(old_frame_path).name
    candidate = frames_dir / frame_name
    if candidate.exists():
        return candidate
    return None


def resolve_video_path(old_video_path: str, videos_dir: Path) -> str:
    """Resolve an old video path to the current videos directory by filename.

    Returns the original path if the video file is not found, so that the
    metadata remains traceable even when the source video is absent.
    """
    video_name = Path(old_video_path).name
    candidate = videos_dir / video_name
    if candidate.exists():
        return str(candidate)
    return old_video_path


def migrate(settings, backup_suffix: str, dry_run: bool) -> None:
    embeddings_dir = settings.embeddings_dir
    backup_dir = embeddings_dir.parent / f"embeddings_{backup_suffix}"

    if not embeddings_dir.exists():
        print(f"Embeddings directory not found: {embeddings_dir}")
        return

    if backup_dir.exists():
        print(f"Backup directory already exists: {backup_dir}")
        print("Remove it or use a different --backup-suffix.")
        return

    # Step 1: 归档旧 embedding
    if not dry_run:
        shutil.move(str(embeddings_dir), str(backup_dir))
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        print(f"Archived old embeddings to {backup_dir}")
    else:
        print(f"[DRY RUN] Would archive {embeddings_dir} -> {backup_dir}")

    # Step 2: 加载 CN-CLIP
    settings.clip_backend = "cnclip"
    service = EmbeddingService(settings)
    service.load_model()
    embedding_dim = service.get_embedding_dim()
    print(f"CN-CLIP loaded. Embedding dim = {embedding_dim}")

    # Step 3: 遍历旧 embedding，重新编码
    json_files = sorted(backup_dir.glob("*.json"))
    total_videos = len(json_files)
    total_frames = 0
    skipped_frames = 0

    for i, json_file in enumerate(json_files, 1):
        metadata_list = json.loads(json_file.read_text(encoding="utf-8"))
        if not metadata_list:
            continue

        video_id = metadata_list[0].get("video_id", json_file.stem)

        # 修正路径：项目目录可能已移动，需将旧路径映射到当前目录
        frame_paths: list[Path] = []
        valid_metadata: list[dict] = []
        for item in metadata_list:
            resolved_frame = resolve_frame_path(item["frame_path"], settings.frames_dir)
            if resolved_frame is None:
                skipped_frames += 1
                continue
            frame_paths.append(resolved_frame)
            resolved_video = resolve_video_path(
                item.get("video_path", ""), settings.videos_dir
            )
            valid_metadata.append(
                {
                    **item,
                    "frame_path": str(resolved_frame),
                    "video_path": resolved_video,
                }
            )

        if not frame_paths:
            print(f"  [{i}/{total_videos}] {video_id}: no valid frames, skipping")
            continue

        if not dry_run:
            # 批量编码（不传 output_name，手动保存完整元数据）
            records = service.batch_encode(
                frame_paths=frame_paths,
                batch_size=8,
            )

            # 保存 .npy（向量矩阵）和 .json（完整元数据）
            npy_path = settings.embeddings_dir / f"{video_id}.npy"
            json_path = settings.embeddings_dir / f"{video_id}.json"
            matrix = np.asarray(
                [r["embedding"] for r in records], dtype=np.float32
            )
            np.save(npy_path, matrix)

            json_records = [
                {k: v for k, v in meta.items() if k != "embedding"}
                for meta in valid_metadata
            ]
            json_path.write_text(
                json.dumps(json_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            total_frames += len(records)
            print(f"  [{i}/{total_videos}] {video_id}: encoded {len(records)} frames")
        else:
            total_frames += len(frame_paths)
            print(
                f"  [DRY RUN] [{i}/{total_videos}] {video_id}: "
                f"would encode {len(frame_paths)} frames"
            )

    print(f"\nMigration {'simulated' if dry_run else 'complete'}:")
    print(f"  Videos processed: {total_videos}")
    print(f"  Frames encoded: {total_frames}")
    print(f"  Frames skipped: {skipped_frames}")

    # Step 4: 重建 FAISS 索引
    if not dry_run:
        print("\nRebuilding FAISS index...")
        index_service = IndexService(settings)
        index_service.rebuild_index()
        stats = index_service.get_stats()
        print(f"  Indexed frames: {stats['indexed_frames']}")
        print(f"  Indexed videos: {stats['indexed_videos']}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate embeddings from OpenCLIP to CN-CLIP"
    )
    parser.add_argument(
        "--backup-suffix",
        default="openclip_backup",
        help="Suffix for the backup directory (default: openclip_backup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without actual encoding",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.clip_backend = "cnclip"

    migrate(settings, args.backup_suffix, args.dry_run)


if __name__ == "__main__":
    main()
