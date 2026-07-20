from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from core.app_state import AppState


SUPPORTED_MODES = ("clip", "detection", "trajectory", "hybrid")


def intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Return True when two time windows overlap."""
    return max(a_start, b_start) <= min(a_end, b_end)


def normalize_video_name(video_name: str) -> str:
    """Normalize a video name so .avi/.mp4 variants of the same asset can compare fairly."""
    return Path(str(video_name)).stem.casefold()


def evaluate_query(results: List[Dict], expected_segments: List[Dict], top_k: int) -> Dict[str, float]:
    """Compute simple Precision@K and hit recall for one query."""
    limited_results = results[:top_k]
    hits = 0
    matched_expected = set()

    for result in limited_results:
        for idx, expected in enumerate(expected_segments):
            if normalize_video_name(result.get("video_name", "")) != normalize_video_name(expected.get("video_name", "")):
                continue
            if intervals_overlap(
                float(result.get("start_ts", 0.0)),
                float(result.get("end_ts", 0.0)),
                float(expected.get("start_ts", 0.0)),
                float(expected.get("end_ts", 0.0)),
            ):
                hits += 1
                matched_expected.add(idx)
                break

    precision_at_k = hits / top_k if top_k > 0 else 0.0
    recall = len(matched_expected) / len(expected_segments) if expected_segments else 0.0
    return {
        "precision_at_k": precision_at_k,
        "recall": recall,
        "hits": hits,
    }


def normalize_clip_results(raw_results: List[Dict]) -> List[Dict]:
    return [
        {
            "video_name": item["video_name"],
            "start_ts": item["timestamp"],
            "end_ts": item["timestamp"],
            "score": item["score"],
        }
        for item in raw_results
    ]


def normalize_detection_results(raw_results) -> List[Dict]:
    return [
        {
            "video_name": item.video_name,
            "start_ts": item.timestamp,
            "end_ts": item.timestamp,
            "score": item.confidence,
            "matched_label": item.matched_label,
        }
        for item in raw_results
    ]


def normalize_trajectory_results(raw_results) -> List[Dict]:
    return [
        {
            "video_name": item.video_name,
            "start_ts": item.start_ts,
            "end_ts": item.end_ts,
            "score": item.avg_confidence,
            "matched_label": item.label,
            "matched_direction": item.direction,
        }
        for item in raw_results
    ]


def collect_mode_results(services, query: str, mode: str, top_k: int) -> List[Dict]:
    """Collect normalized results for one retrieval mode."""
    if mode == "clip":
        query_variants = services.query_rewrite_service.rewrite_query(query)
        clip_query = query_variants[0] if query_variants else query
        raw_results = services.search_service.search_text(query=clip_query, top_k=top_k)
        return normalize_clip_results(raw_results)

    if mode == "detection":
        raw_results = services.detection_search_service.search_objects(query=query, top_k=top_k)
        return normalize_detection_results(raw_results)

    if mode == "trajectory":
        raw_results = services.trajectory_search_service.search_tracks(query=query, top_k=top_k)
        return normalize_trajectory_results(raw_results)

    if mode == "hybrid":
        response = services.hybrid_search_service.search(query=query, top_k=top_k)
        return list(response.get("results", []))

    raise ValueError(f"Unsupported eval mode: {mode}")


def resolve_modes(query_item: Dict) -> List[str]:
    """Resolve which retrieval modes to evaluate for a query item."""
    explicit_modes = query_item.get("modes")
    if explicit_modes:
        return [str(mode).lower() for mode in explicit_modes if str(mode).lower() in SUPPORTED_MODES]

    mode = str(query_item.get("mode", "hybrid")).lower()
    if mode == "all":
        return list(SUPPORTED_MODES)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported eval mode: {mode}")
    return [mode]


def run_eval(eval_config_path: str | Path) -> Dict[str, object]:
    """Run offline traffic retrieval evaluation from a JSON template."""
    config_path = Path(eval_config_path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    settings = get_settings()
    services = AppState.build(settings)

    query_reports = []
    mode_accumulators: Dict[str, Dict[str, float]] = {
        mode: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0.0}
        for mode in SUPPORTED_MODES
    }

    for item in payload.get("queries", []):
        query = str(item["query"])
        top_k = int(item.get("top_k", 5))
        expected_segments = list(item.get("expected_segments", []))
        modes = resolve_modes(item)

        mode_reports = {}
        for mode in modes:
            results = collect_mode_results(services=services, query=query, mode=mode, top_k=top_k)
            metrics = evaluate_query(results=results, expected_segments=expected_segments, top_k=top_k)
            mode_reports[mode] = {
                "metrics": metrics,
                "result_count": len(results),
            }
            mode_accumulators[mode]["precision_sum"] += metrics["precision_at_k"]
            mode_accumulators[mode]["recall_sum"] += metrics["recall"]
            mode_accumulators[mode]["count"] += 1

        query_reports.append(
            {
                "query": query,
                "top_k": top_k,
                "expected_segments": expected_segments,
                "mode_reports": mode_reports,
            }
        )

    per_mode_summary = {}
    for mode, accumulator in mode_accumulators.items():
        if accumulator["count"] == 0:
            continue
        per_mode_summary[mode] = {
            "macro_precision_at_k": accumulator["precision_sum"] / accumulator["count"],
            "macro_recall": accumulator["recall_sum"] / accumulator["count"],
            "query_count": int(accumulator["count"]),
        }

    return {
        "dataset_name": payload.get("dataset_name", ""),
        "per_mode_summary": per_mode_summary,
        "query_reports": query_reports,
    }


if __name__ == "__main__":
    default_path = Path(__file__).with_name("traffic_retrieval_eval_template.json")
    report = run_eval(default_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
