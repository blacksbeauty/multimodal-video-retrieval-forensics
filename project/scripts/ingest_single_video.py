# -*- coding: utf-8 -*-
"""单视频一键入库（完整管线）：复制 → 抽帧+CLIP+FAISS帧索引 → YOLO检测(仅新帧)
→ ByteTrack轨迹 → 事件插件 → 段级索引。逐步打印每步结果。

用法（在 project/ 目录下执行）:
    python scripts/ingest_single_video.py "D:\\视频路径.mp4" [--name 目标文件名] [--frame-interval 12] [--skip-event]

示例:
    python scripts/ingest_single_video.py "D:\\生成道路监控闯红灯视频 (2).mp4"
    python scripts/ingest_single_video.py "D:\\xxx.mp4" --name 我的视频.mp4 --frame-interval 12
"""
import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from config import Settings
from core.schemas import DetectionVideoMetadata
from services.detection_service import DetectionService
from services.embedding_service import EmbeddingService
from services.event_service import EventService
from services.index_service import IndexService
from services.segment_build_service import SegmentBuildService
from services.segment_service import SegmentService
from services.tracking_service import TrackingService
from services.video_service import VideoService
from utils.path_utils import build_asset_id, normalize_path


def main() -> int:
    parser = argparse.ArgumentParser(description="单视频一键入库（完整摄取管线）")
    parser.add_argument("video", help="源视频绝对路径")
    parser.add_argument("--name", default=None, help="复制到 videos/ 后的目标文件名（默认保留源文件名）")
    parser.add_argument("--frame-interval", type=int, default=12, help="抽帧间隔（24fps 下 12≈2帧/秒，短视频建议 12，长视频可用默认 30）")
    parser.add_argument("--skip-event", action="store_true", help="跳过事件识别与段构建（仅索引+检测+轨迹）")
    args = parser.parse_args()

    settings = Settings()
    source = Path(args.video)
    if not source.is_file():
        print(f"[错误] 源文件不存在: {source}")
        return 1

    dst_name = args.name or source.name
    dst_path = settings.videos_dir / dst_name
    settings.videos_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[步骤 0] 复制视频到 videos/")
    if dst_path.exists():
        print(f"  - 已存在（跳过复制）: {dst_name}")
    else:
        shutil.copy2(source, dst_path)
        print(f"  ✓ 复制完成: {dst_name} ({source.stat().st_size / 1024:.0f} KB)")

    clip_service = EmbeddingService(settings)
    index_service = IndexService(settings)
    video_service = VideoService(settings)
    detection_service = DetectionService(settings)
    tracking_service = TrackingService(settings)
    event_service = EventService(settings)
    segment_build_service = SegmentBuildService(settings, SegmentService(settings))

    video_id = build_asset_id(dst_path)
    print("=" * 60)
    print(f"处理视频: {dst_name}  (video_id={video_id})")

    # 步骤 1：抽帧 + CLIP 编码 + 帧级 FAISS 索引
    print(f"[步骤 1] 抽帧(间隔={args.frame_interval}) + CLIP 编码 + FAISS 增量索引 ...")
    ingest = video_service.ingest_video(
        video_path=str(dst_path),
        frame_interval=args.frame_interval,
        clip_service=clip_service,
        index_service=index_service,
    )
    print(f"   ✓ 总帧数={ingest['total_frames']} 抽取帧数={ingest['extracted_frames']} 已索引={ingest['indexed_frames']}")

    # 步骤 2：YOLO 检测（仅本视频新帧）
    print("[步骤 2] YOLO 目标检测（仅本视频帧）...")
    frame_paths = sorted(settings.frames_dir.glob(f"*{Path(dst_name).stem}_*.jpg"))
    if not frame_paths:
        print("   !! 未找到新帧，跳过检测")
        return 1
    grouped: dict[str, list] = defaultdict(list)
    for frame_path in frame_paths:
        meta = detection_service.detect_frame(frame_path)
        if meta is not None:
            grouped[meta.video_name].append(meta)
    for video_name, frame_metas in grouped.items():
        resolved_video_path = detection_service._resolve_video_path_from_frame_group(video_name)
        v_id = build_asset_id(resolved_video_path)
        ordered = sorted(frame_metas, key=lambda item: item.timestamp)
        for frame in ordered:
            frame.video_name = resolved_video_path.name
        detection_service.save_detection_metadata(
            DetectionVideoMetadata(
                video_id=v_id,
                video_name=resolved_video_path.name,
                video_path=normalize_path(resolved_video_path),
                frames=ordered,
            )
        )
        det_count = sum(len(f.detections) for f in ordered)
        labels = sorted({d.label for f in ordered for d in f.detections})
        print(f"   ✓ 检测保存 video_id={v_id} 帧数={len(ordered)} 框={det_count} 类别={labels}")

    # 步骤 3：轨迹
    print("[步骤 3] ByteTrack 轨迹构建 ...")
    det_path = settings.detection_metadata_dir / f"{video_id}.json"
    trajectory_meta = tracking_service.process_video_metadata(det_path)
    if trajectory_meta is None or not trajectory_meta.tracks:
        print("   - 无轨迹（未检出可跟踪目标）")
    else:
        tracking_service.save_trajectory_metadata(trajectory_meta)
        total_points = sum(len(t.points) for t in trajectory_meta.tracks)
        dirs = sorted({t.direction for t in trajectory_meta.tracks})
        print(f"   ✓ 轨迹保存 tracks={len(trajectory_meta.tracks)} points={total_points} 方向={dirs}")

    # 步骤 4/5：事件 + 段级索引
    if args.skip_event:
        print("[步骤 4/5] 已跳过（--skip-event）")
    else:
        print("[步骤 4] 事件插件识别（红绿灯/压线/逆行）...")
        traj_path = settings.trajectory_metadata_dir / f"{video_id}.json"
        event_bundle = event_service.process_video_metadata(
            trajectory_path=traj_path,
            detection_dir=str(settings.detection_metadata_dir),
            plugin_names=None,
        )
        if event_bundle is None or not event_bundle.events:
            print("   - 无事件命中")
        else:
            event_service.save_event_metadata(event_bundle)
            ev_types = sorted({e.event_type for e in event_bundle.events})
            print(f"   ✓ 事件保存 事件数={len(event_bundle.events)} 类型={ev_types}")

        print("[步骤 5] 段级文本编码 + 段级 FAISS upsert ...")
        seg = segment_build_service.ingest_segment_pipeline(
            video_id=video_id, clip_service=clip_service, index_service=index_service,
        )
        print(f"   {'✓' if seg.get('built') else '✗'} 段构建: {seg}")

    print("=" * 60)
    print(f"[完成] {dst_name}  video_id={video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
