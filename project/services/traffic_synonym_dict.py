"""Traffic event semantic synonym dictionary.

Provides expanded event aliases, attribute patterns, and inference rules
for the Traffic Query Understanding Module (TQUM).

This module replaces the narrow _EVENT_ALIASES and _EXACT_REWRITE_RULES
in query_rewrite_service.py with a multi-tier semantic matching system:

  Layer 1 — direct / colloquial / descriptive / implicit event aliases
  Layer 2 — implicit event inference rules (keyword combination → event)
  Layer 3 — expanded attribute patterns (color, vehicle_type, light_state)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Layer 1: Event semantic clusters
# ---------------------------------------------------------------------------
# Each event type has four semantic tiers:
#   direct      — exact domain terms (confidence: 1.0)
#   descriptive — formal descriptive phrases (confidence: 0.8)
#   colloquial  — everyday spoken expressions (confidence: 0.7)
#   implicit    — indirect/abstract references (confidence: 0.6)

EVENT_SEMANTIC_CLUSTERS: Dict[str, Dict[str, List[str]]] = {
    "wrong_way_driving": {
        "direct": [
            "逆行", "逆向行驶", "反向行驶", "反方向行驶", "逆方向行驶",
            "机动车逆行", "车辆逆行", "汽车逆行",
        ],
        "descriptive": [
            "朝错误方向行驶", "朝相反方向移动", "朝相反方向行驶",
            "车辆方向与道路规定相反", "违反道路行驶方向",
            "驶入对向车道", "进入反向道路", "进入对向车道",
            "违规反向行驶", "不按规定方向行驶",
            "反向驶入道路", "逆向通过道路",
        ],
        "colloquial": [
            "开反了", "走反方向", "开反方向的车", "走错方向", "开错方向",
            "反着开", "逆着开", "逆着车流行驶", "对着开", "反着来",
            "不按规定路线走", "不按规定路线行驶",
        ],
        "implicit": [
            "方向异常", "方向不对", "走错路", "开错路",
            "方向与道路规定相反", "错误方向", "相反方向", "反方向",
            "对向", "反向",
        ],
    },
    "red_light_violation": {
        "direct": [
            "闯红灯", "冲红灯", "红灯违规", "红灯违章", "红灯违法",
            "车辆闯红灯", "机动车闯红灯",
        ],
        "descriptive": [
            "红灯时通过路口", "红灯亮时通行", "红灯亮时车辆通过路口",
            "红灯状态下继续行驶", "红灯期间通过交叉路口",
            "违反交通信号灯", "未遵守红绿灯规则",
            "不按交通信号通行", "红灯时通过",
            "禁止通行时间通过路口", "信号灯违规",
        ],
        "colloquial": [
            "抢红灯", "不守红绿灯", "不守信号灯", "闯信号灯",
            "冲信号灯", "无视红灯", "忽略红灯", "不等红灯",
            "不等红灯继续行驶", "没有等待红灯",
            "小车冲过红灯", "冲过红灯",
        ],
        "implicit": [
            "红灯时", "红灯期间", "红灯亮", "红灯状态",
            "违反信号", "不守灯", "闯灯",
        ],
    },
    "vehicle_crosses_line": {
        "direct": [
            "压线", "压线行驶", "跨线", "跨线行驶", "越线行驶",
            "车辆压线", "货车压线",
        ],
        "descriptive": [
            "越过车道线", "跨越道路标线", "驶过车道边界",
            "偏离正常车道", "越过分界线", "占用其他车道",
            "未保持在车道内", "压道路实线", "压道路标线",
            "越过道路标线行驶", "压着道路标线开",
        ],
        "colloquial": [
            "压到线", "压着线开", "压着标线开", "越过线了",
            "没有保持车道", "没守车道", "跑偏了",
            "大车压到线", "货车压到线",
        ],
        "implicit": [
            "车道偏离", "不按车道行驶", "偏离车道",
            "没有保持在车道", "压道路",
        ],
    },
}


# Flattened lookup: alias → (event_type, confidence_tier)
EVENT_ALIAS_INDEX: Dict[str, Tuple[str, float]] = {}

_TIER_CONFIDENCE = {"direct": 1.0, "descriptive": 0.8, "colloquial": 0.7, "implicit": 0.6}


def _build_alias_index() -> None:
    for event_type, tiers in EVENT_SEMANTIC_CLUSTERS.items():
        for tier, aliases in tiers.items():
            confidence = _TIER_CONFIDENCE[tier]
            for alias in aliases:
                # Shorter aliases only set if not already set by a higher tier
                existing = EVENT_ALIAS_INDEX.get(alias)
                if existing is None or existing[1] < confidence:
                    EVENT_ALIAS_INDEX[alias] = (event_type, confidence)


_build_alias_index()


# ---------------------------------------------------------------------------
# Layer 2: Implicit event inference rules
# ---------------------------------------------------------------------------
# When Layer 1 doesn't directly match, infer event type from keyword combos.

INFERENCE_RULES: List[Dict[str, Any]] = [
    {
        "event_type": "wrong_way_driving",
        "confidence": 0.7,
        "conditions": [
            {
                "any_of": ["错误方向", "相反方向", "反方向", "对向", "反向",
                           "走错", "开错", "逆着", "反着", "对着开"],
                "context_required": ["道路", "车道", "行驶", "驾驶", "开", "车",
                                      "通过", "驶入", "进入"],
            },
        ],
    },
    {
        "event_type": "red_light_violation",
        "confidence": 0.7,
        "conditions": [
            {
                "all_of": ["红灯"],
                "any_of": ["通过", "通行", "行驶", "继续", "冲", "抢",
                           "忽略", "无视", "不等", "没有等待", "闯"],
            },
        ],
    },
    {
        "event_type": "vehicle_crosses_line",
        "confidence": 0.7,
        "conditions": [
            {
                "any_of": ["偏离", "越过", "跨越", "占用", "没有保持", "压"],
                "context_required": ["车道", "标线", "实线", "分界线", "边界", "线"],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Layer 3: Expanded attribute patterns
# ---------------------------------------------------------------------------

ATTRIBUTE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "color": {
        "white": ["白色", "白车", "浅色", "颜色较浅", "较浅", "白"],
        "red": ["红色", "红车"],
        "black": ["黑色", "黑车"],
        "blue": ["蓝色", "蓝车"],
        "yellow": ["黄色", "黄车"],
    },
    "vehicle_type": {
        "car": ["汽车", "轿车", "小汽车", "小车", "机动车", "轿车"],
        "truck": ["货车", "卡车", "大车", "大货车", "大型车辆"],
        "bus": ["公交车", "巴士", "大巴"],
        "motorcycle": ["摩托车", "摩托"],
    },
    "light_state": {
        "red": ["红灯", "红灯亮", "红灯状态", "红灯时", "红灯期间"],
        "green": ["绿灯"],
        "yellow": ["黄灯"],
    },
}


# Flattened: pattern → (attribute_category, attribute_value)
_ATTRIBUTE_INDEX: Dict[str, Tuple[str, str]] = {}


def _build_attribute_index() -> None:
    for category, values in ATTRIBUTE_PATTERNS.items():
        for value, patterns in values.items():
            for pattern in patterns:
                existing = _ATTRIBUTE_INDEX.get(pattern)
                # Prefer more specific (longer) patterns
                if existing is None or len(pattern) > len(existing[1]):
                    _ATTRIBUTE_INDEX[pattern] = (category, value)


_build_attribute_index()


# ---------------------------------------------------------------------------
# Chinese display names for event types
# ---------------------------------------------------------------------------

EVENT_CN_DISPLAY: Dict[str, str] = {
    "wrong_way_driving": "逆行",
    "red_light_violation": "闯红灯",
    "vehicle_crosses_line": "压线",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_event_type(query: str) -> Tuple[str, float]:
    """Match a query against the event semantic clusters.

    Returns (event_type, confidence) or ("", 0.0) if no match.
    Uses longest-match-first to avoid partial collisions.
    """
    best_type = ""
    best_conf = 0.0

    # Sort aliases by length descending for longest-match-first
    for alias, (event_type, confidence) in sorted(
        EVENT_ALIAS_INDEX.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if alias in query and confidence > best_conf:
            best_type = event_type
            best_conf = confidence

    return best_type, best_conf


def match_all_events(query: str) -> List[Tuple[str, float]]:
    """Match all event types present in the query.

    Returns list of (event_type, confidence), sorted by confidence descending.
    """
    results: Dict[str, float] = {}
    for alias, (event_type, confidence) in EVENT_ALIAS_INDEX.items():
        if alias in query:
            existing = results.get(event_type, 0.0)
            if confidence > existing:
                results[event_type] = confidence
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def infer_event_types(
    query: str,
    primary_entities: List[str],
    context_entities: List[str],
) -> List[Tuple[str, float]]:
    """Infer event types from keyword combinations when no direct match exists.

    Returns list of (event_type, confidence) from inference rules.
    """
    inferred: Dict[str, float] = {}

    for rule in INFERENCE_RULES:
        for condition in rule["conditions"]:
            any_of = condition.get("any_of", [])
            all_of = condition.get("all_of", [])
            context_required = condition.get("context_required", [])

            # Check any_of (at least one must match)
            any_matched = False
            if any_of:
                for keyword in any_of:
                    if keyword in query:
                        any_matched = True
                        break
                if not any_matched:
                    continue
            else:
                any_matched = True

            # Check all_of (all must match)
            all_matched = True
            for keyword in all_of:
                if keyword not in query:
                    all_matched = False
                    break
            if not all_matched:
                continue

            # Check context
            context_matched = True
            if context_required:
                context_matched = any(ctx in query for ctx in context_required)
                # Also accept if entity detection found context
                if not context_matched:
                    entity_strs = " ".join(primary_entities + context_entities)
                    context_matched = any(ctx in entity_strs for ctx in context_required)

            if any_matched and all_matched and context_matched:
                event_type = rule["event_type"]
                conf = rule["confidence"]
                existing = inferred.get(event_type, 0.0)
                if conf > existing:
                    inferred[event_type] = conf

    return sorted(inferred.items(), key=lambda x: x[1], reverse=True)


def extract_attributes(query: str) -> Dict[str, Any]:
    """Extract traffic attributes from a Chinese query.

    Returns dict with optional keys: color, vehicle_type, light_state.
    """
    attributes: Dict[str, Any] = {}

    # Color detection (check longer patterns first to avoid "白" matching before "白色")
    for pattern, (category, value) in sorted(
        _ATTRIBUTE_INDEX.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if category == "color" and pattern in query:
            if "color" not in attributes:
                attributes["color"] = value

    # Vehicle type
    for pattern, (category, value) in sorted(
        _ATTRIBUTE_INDEX.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if category == "vehicle_type" and pattern in query:
            if "vehicle_type" not in attributes:
                attributes["vehicle_type"] = value

    # Light state (exclude "红绿灯"/"信号灯" which are device names, not states)
    if "红绿灯" not in query and "信号灯" not in query:
        for pattern, (category, value) in sorted(
            _ATTRIBUTE_INDEX.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if category == "light_state" and pattern in query:
                if "light_state" not in attributes:
                    attributes["light_state"] = value

    return attributes
