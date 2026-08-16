# -*- coding: utf-8 -*-
"""一次性迁移脚本：旧 event_id 格式 → 新格式（数据规范 v1.1 §2 ID 规则）。

旧格式: {video_id}:{event_type}:{video_id}:{n}   （track_id 含 video_id 前缀，导致冗余）
新格式: {video_id}:{event_type}:{n}              （v1.1 定稿，去冗余 + 支持多对象事件）

注意:
- 旧格式第三段实际是 track_id（形如 {video_id}:{n}），本脚本将整体拆分为
  [video_id, event_type, video_id, n]，去掉重复的 video_id 段，保留尾部序号。
- 幂等：已符合新格式的 event_id 原样保留。
- 只迁移 metadata/events/*.json；segment/检索引用旧 ID 的，需按迁移映射表更新。

用法:
    python tests/migrate_event_ids.py [--dry-run]

--dry-run 只打印将发生的变更，不写文件。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_event_id(event_id: str) -> str | None:
    """旧 → 新 event_id。已是新格式返回 None（无需变更）。"""
    parts = event_id.split(":")
    # 新格式恰好 3 段: {video_id}:{event_type}:{n}
    if len(parts) == 3:
        return None
    # 旧格式 4 段: {video_id}:{event_type}:{video_id}:{n}
    if len(parts) == 4 and parts[0] == parts[2]:
        return f"{parts[0]}:{parts[1]}:{parts[3]}"
    logger.warning("无法识别的 event_id 格式，保持原样: %s", event_id)
    return None


def migrate_file(path: Path, dry_run: bool) -> dict:
    """迁移单个 events 文件，返回统计。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    changed = 0
    mapping: dict[str, str] = {}  # old -> new，供下游 segment 引用更新
    for event in events:
        old_id = event.get("event_id", "")
        new_id = migrate_event_id(old_id)
        if new_id is not None:
            mapping[old_id] = new_id
            event["event_id"] = new_id
            changed += 1

    if changed and not dry_run:
        # 原子写：临时文件 + replace（与 detection_service 的写入模式一致）
        tmp = path.with_name(f"{path.name}.{id(path)}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    return {"path": str(path), "changed": changed, "mapping": mapping}


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy event_id format to v1.1.")
    parser.add_argument("--dry-run", action="store_true", help="仅预览变更，不写文件")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    events_dir = project_root / "metadata" / "events"
    if not events_dir.exists():
        logger.error("events 目录不存在: %s", events_dir)
        return 1

    total_changed = 0
    total_mapping: dict[str, str] = {}
    for path in sorted(events_dir.glob("*.json")):
        stat = migrate_file(path, dry_run=args.dry_run)
        total_changed += stat["changed"]
        total_mapping.update(stat["mapping"])
        if stat["changed"]:
            action = "DRY-RUN 预览" if args.dry_run else "已迁移"
            logger.info("%s: %s（%s 条 event_id 变更）", action, stat["path"], stat["changed"])

    logger.info(
        "合计 %s 条 event_id 迁移（%s），涉及文件 %s 个",
        total_changed,
        "dry-run 未写盘" if args.dry_run else "已写盘",
        len(total_mapping),
    )

    if total_mapping and not args.dry_run:
        # 输出映射表供 segment/检索引用更新
        out_path = project_root / "metadata" / "event_id_migration_map.json"
        out_path.write_text(
            json.dumps(total_mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("迁移映射表已写入: %s（segment 中引用旧 event_id 的需同步更新）", out_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
