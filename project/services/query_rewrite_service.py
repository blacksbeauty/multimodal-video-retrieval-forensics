from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from core.schemas import QueryIntent
from services.traffic_synonym_dict import (
    EVENT_CN_DISPLAY,
    match_all_events,
    match_event_type,
    infer_event_types,
    extract_attributes as extract_tqum_attributes,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AliasRule:
    """One deterministic alias rule in the traffic ontology."""

    aliases: tuple[str, ...]
    entities: tuple[str, ...]
    group: str
    role: str


TRAFFIC_ONTOLOGY: Dict[str, Any] = {
    "entities": {
        "car": {"group": "vehicle"},
        "truck": {"group": "vehicle"},
        "bus": {"group": "vehicle"},
        "motorcycle": {"group": "vehicle"},
        "person": {"group": "person"},
        "traffic_light": {"group": "traffic"},
        "stop_line": {"group": "traffic"},
        "lane": {"group": "traffic"},
        "intersection": {"group": "traffic"},
        "crosswalk": {"group": "traffic"},
    },
    "relations": {
        "near": ("near", "\u9644\u8fd1", "\u9760\u8fd1"),
        "inside": ("inside", "\u5185", "\u5728.*\u91cc"),
        "cross": ("cross", "crossing", "\u7a7f\u8fc7", "\u8de8\u8fc7", "\u8d8a\u8fc7"),
        "entering": ("entering", "enter", "\u8fdb\u5165"),
        "leaving": ("leaving", "leave", "\u79bb\u5f00"),
        "left_of": ("left of", "\u5de6\u4fa7", "\u5de6\u8fb9"),
        "right_of": ("right of", "\u53f3\u4fa7", "\u53f3\u8fb9"),
    },
    "directions": {
        "left_to_right": ("left to right", "\u4ece\u5de6\u5230\u53f3", "\u5411\u53f3"),
        "right_to_left": ("right to left", "\u4ece\u53f3\u5230\u5de6", "\u5411\u5de6"),
        "top_to_bottom": ("top to bottom", "\u4ece\u4e0a\u5230\u4e0b"),
        "bottom_to_top": ("bottom to top", "\u4ece\u4e0b\u5230\u4e0a"),
    },
    "motions": {
        "turn_left": ("turn left", "\u5de6\u8f6c"),
        "turn_right": ("turn right", "\u53f3\u8f6c"),
        "stop": ("stop", "\u505c\u8f66", "\u505c\u4f4f"),
        "park": ("park", "\u505c\u9760", "\u505c\u653e"),
    },
    "attributes": {
        "light_state": {
            "red": ("red light", "\u7ea2\u706f"),
            "yellow": ("yellow light", "\u9ec4\u706f"),
            "green": ("green light", "\u7eff\u706f"),
        },
        "duration": {
            "long": ("\u957f\u65f6\u95f4", "long time", "for a long time"),
        },
        "color": {
            "white": ("white", "\u767d\u8272"),
            "black": ("black", "\u9ed1\u8272"),
        },
    },
}

_ALIAS_RULES: tuple[AliasRule, ...] = (
    AliasRule(("\u767d\u8272\u6c7d\u8f66",), ("car",), "vehicle", "primary"),
    AliasRule(("\u767d\u8272\u8f7f\u8f66",), ("car",), "vehicle", "primary"),
    AliasRule(("\u8f66\u724c",), tuple(), "context", "context"),
    AliasRule(("traffic light", "\u7ea2\u7eff\u706f", "\u4fe1\u53f7\u706f"), ("traffic_light",), "traffic", "context"),
    AliasRule(("stop line", "\u505c\u6b62\u7ebf"), ("stop_line",), "traffic", "context"),
    AliasRule(("intersection", "\u8def\u53e3", "\u4ea4\u53c9\u53e3"), ("intersection",), "traffic", "context"),
    AliasRule(("crosswalk", "\u6591\u9a6c\u7ebf", "\u4eba\u884c\u9053"), ("crosswalk",), "traffic", "context"),
    AliasRule(("lane", "\u8f66\u9053"), ("lane",), "traffic", "context"),
    AliasRule(("truck", "\u8d27\u8f66", "\u5361\u8f66"), ("truck",), "vehicle", "primary"),
    AliasRule(("bus", "\u516c\u4ea4\u8f66", "\u5df4\u58eb"), ("bus",), "vehicle", "primary"),
    AliasRule(("motorcycle", "\u6469\u6258\u8f66", "\u6469\u6258"), ("motorcycle",), "vehicle", "primary"),
    AliasRule(("person", "\u884c\u4eba"), ("person",), "person", "primary"),
    AliasRule(("car", "\u6c7d\u8f66", "\u8f7f\u8f66", "\u5c0f\u6c7d\u8f66", "\u8f66"), ("car",), "vehicle", "primary"),
    AliasRule(("vehicle", "vehicles", "\u8f66\u8f86"), ("car", "truck", "bus", "motorcycle"), "vehicle", "primary"),
)

_EXACT_REWRITE_RULES: Dict[str, List[str]] = {
    "\u767d\u8272\u6c7d\u8f66": ["white car", "white sedan", "white vehicle", "car"],
    "\u767d\u8272\u8f7f\u8f66": ["white sedan", "white car", "white vehicle", "car"],
    "\u9ed1\u8863\u7537\u5b50": ["man in black", "person wearing black clothes"],
    "\u8f66\u724c": ["license plate", "vehicle plate"],
    "\u6c7d\u8f66": ["car"],
    "\u8f7f\u8f66": ["car"],
    "\u8d27\u8f66": ["truck"],
    "\u5361\u8f66": ["truck"],
    "\u516c\u4ea4\u8f66": ["bus"],
    "\u6469\u6258\u8f66": ["motorcycle"],
    "\u7ea2\u7eff\u706f": ["traffic light"],
    "\u884c\u4eba": ["person"],
    "\u8f66\u8f86": ["car", "truck", "bus", "motorcycle"],
}

_PRIMARY_GROUPS = {"vehicle", "person"}
_SEMANTIC_TERMS = ("white", "black", "\u767d\u8272", "\u9ed1\u8272")
_EVENT_ALIASES = {
    "vehicle_crosses_line": ("vehicle_crosses_line", "cross line", "line crossing", "压线", "车辆压线", "压线行驶", "跨线", "跨线行驶"),
    "wrong_way_driving": ("wrong_way_driving", "wrong way driving", "wrong way", "逆行", "车辆逆行", "反向行驶", "逆向行驶", "逆方向行驶", "反方向", "opposite direction"),
    "red_light_violation": ("red_light_violation", "red light violation", "闯红灯", "车辆闯红灯", "冲红灯", "红灯违规", "红灯违章", "红灯违法", "违反红灯", "running red light"),
}

_EVENT_SYNONYMS: Dict[str, List[str]] = {
    "逆行": ["车辆逆行", "车辆逆向行驶", "机动车逆行", "车辆反方向行驶", "逆向行驶"],
    "闯红灯": ["车辆闯红灯", "冲红灯", "红灯违规", "红灯违章"],
    "压线": ["车辆压线", "压线行驶", "跨线行驶", "车辆越线"],
}

_ENTITY_CN_DISPLAY: Dict[str, str] = {
    "car": "汽车",
    "truck": "货车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "person": "行人",
    "traffic_light": "交通信号灯",
    "stop_line": "停止线",
    "intersection": "路口",
    "crosswalk": "斑马线",
    "lane": "车道",
}


class QueryRewriteService:
    """Parse deterministic multi-entity traffic intents and derive retrieval-friendly rewrites."""

    def __init__(self, use_chinese_clip: bool = False) -> None:
        self.ontology = TRAFFIC_ONTOLOGY
        self.use_chinese_clip = use_chinese_clip

    def rewrite_query(self, query: str) -> List[str]:
        """Expand a query into retrieval-friendly variants while preserving backward compatibility."""
        original_query = query.strip()
        if not original_query:
            return []

        if self._is_english_query(original_query):
            return [original_query]

        exact = _EXACT_REWRITE_RULES.get(original_query)
        if exact is not None:
            return list(exact)

        intent = self.parse_query_intent(original_query)
        return list(intent.rewritten_queries)

    def expand_query(self, query: str) -> List[str]:
        """Expand a Chinese query with traffic-domain synonyms for CN-CLIP retrieval."""
        expansions = [query]
        for keyword, synonyms in _EVENT_SYNONYMS.items():
            if keyword in query:
                for synonym in synonyms:
                    if synonym not in expansions:
                        expansions.append(synonym)
        return expansions

    def normalize_label_query(self, query: str) -> str:
        """Normalize a free-form traffic query fragment for deterministic matching."""
        return re.sub(r"\s+", " ", query.strip().casefold())

    def parse_query_intent(self, query: str) -> QueryIntent:
        """Parse a traffic short query into a multi-entity intent model.

        TQUM integration: uses expanded synonym matching (Layer 1),
        implicit event inference (Layer 2), and expanded attribute extraction (Layer 3).
        """
        normalized_query = self.normalize_label_query(query)
        if not normalized_query:
            return QueryIntent()

        mentions = self._detect_entity_mentions(normalized_query)
        primary_entities, context_entities = self._split_entities(mentions)
        relations = self._detect_named_values(normalized_query, self.ontology["relations"])
        directions = self._detect_named_values(normalized_query, self.ontology["directions"])
        motions = self._detect_named_values(normalized_query, self.ontology["motions"])

        # Layer 1: Expanded event synonym matching from TQUM dictionary
        direct_events = match_all_events(normalized_query)
        event_types = [et for et, _ in direct_events]

        # Layer 3: Expanded attribute extraction from TQUM
        tqum_attributes = extract_tqum_attributes(normalized_query)

        # Legacy attribute detection (keep for backward compatibility)
        legacy_attributes = self._detect_attributes(normalized_query)

        # Merge: TQUM attributes take priority, then legacy
        attributes = {**legacy_attributes, **tqum_attributes}

        if "light_state" in attributes and "traffic_light" not in context_entities and "traffic_light" not in primary_entities:
            context_entities = ["traffic_light", *context_entities]

        # Layer 2: Implicit event inference when no direct match
        if not event_types:
            inferred = infer_event_types(
                normalized_query, primary_entities, context_entities,
            )
            event_types = [et for et, _ in inferred]
            direct_events.extend(inferred)

        # Legacy inference (keep for backward compatibility)
        inferred_event_types = self._infer_event_types(
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            motions=motions,
            attributes=attributes,
            normalized_query=normalized_query,
        )
        for et in inferred_event_types:
            if et not in event_types:
                event_types.append(et)
                direct_events.append((et, 0.5))

        # Calculate event confidence: highest confidence among matched events
        event_confidence = max((conf for _, conf in direct_events), default=0.0)

        query_type = self._classify_query_type(
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            directions=directions,
            motions=motions,
            event_types=event_types,
            attributes=attributes,
        )

        # Phase 3: Build Top-k intent candidates for confidence-based routing
        intent_candidates = self._build_intent_candidates(
            event_types=event_types,
            event_confidence=round(event_confidence, 2),
            primary_entities=primary_entities,
            attributes=attributes,
            query_type=query_type,
            normalized_query=normalized_query,
        )

        # Layer 5: Build Chinese query expansions (not English translations)
        rewritten_queries = self._build_rewrites(
            original_query=query.strip(),
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            directions=directions,
            motions=motions,
            event_types=event_types,
            attributes=attributes,
        )

        intent = QueryIntent(
            query_type=query_type,
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            directions=directions,
            motions=motions,
            event_types=event_types,
            event_confidence=round(event_confidence, 2),
            attributes=attributes,
            rewritten_queries=rewritten_queries,
            normalized_query=normalized_query,
            intent_candidates=intent_candidates,
        )
        logger.info("Parsed query intent query=%s intent=%s", query, intent.model_dump())
        return intent

    def _detect_entity_mentions(self, normalized_query: str) -> List[Dict[str, Any]]:
        raw_mentions: List[Dict[str, Any]] = []
        for rule in _ALIAS_RULES:
            for alias in rule.aliases:
                normalized_alias = self.normalize_label_query(alias)
                for start in self._find_all(normalized_query, normalized_alias):
                    raw_mentions.append(
                        {
                            "start": start,
                            "end": start + len(normalized_alias),
                            "alias": alias,
                            "entities": list(rule.entities),
                            "group": rule.group,
                            "role": rule.role,
                        }
                    )

        accepted: List[Dict[str, Any]] = []
        for mention in sorted(raw_mentions, key=lambda item: (item["start"], -(item["end"] - item["start"]))):
            if any(not (mention["end"] <= existing["start"] or mention["start"] >= existing["end"]) for existing in accepted):
                continue
            accepted.append(mention)

        return accepted

    def _split_entities(self, mentions: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        movable_mentions = [mention for mention in mentions if mention["group"] in _PRIMARY_GROUPS]
        context_mentions = [mention for mention in mentions if mention["group"] not in _PRIMARY_GROUPS]

        primary_entities: List[str] = []
        context_entities: List[str] = []

        if movable_mentions:
            primary_entities = self._dedupe(movable_mentions[-1]["entities"])
            for mention in movable_mentions[:-1]:
                for entity in mention["entities"]:
                    if entity not in primary_entities and entity not in context_entities:
                        context_entities.append(entity)

        for mention in context_mentions:
            for entity in mention["entities"]:
                if entity not in primary_entities and entity not in context_entities:
                    context_entities.append(entity)

        if not primary_entities and context_entities:
            primary_entities = list(context_entities)
            context_entities = []

        return primary_entities, context_entities

    def _classify_query_type(
        self,
        primary_entities: List[str],
        context_entities: List[str],
        relations: List[str],
        directions: List[str],
        motions: List[str],
        event_types: List[str],
        attributes: Dict[str, Any],
    ) -> str:
        if event_types:
            return "event"
        if context_entities and (directions or motions or attributes):
            return "composite"
        if context_entities and relations:
            return "relational"
        if directions or motions:
            return "motion"
        if primary_entities:
            return "object"
        if attributes:
            return "attribute"
        return "general"

    def _build_intent_candidates(
        self,
        event_types: List[str],
        event_confidence: float,
        primary_entities: List[str],
        attributes: Dict[str, Any],
        query_type: str,
        normalized_query: str,
    ) -> List[Dict[str, Any]]:
        """Build Top-k intent candidates with confidence scores.

        Phase 3 optimization: Replaces single-label query_type classification
        with Top-k intent candidates, enabling flexible routing for vague queries.

        Generates up to 3 candidates:
        1. Event intent (if event types detected)
        2. Object intent (if entities detected)
        3. Vague violation intent (fallback for fuzzy queries)
        """
        candidates = []

        # Candidate 1: Event intent (if event types detected)
        if event_types:
            candidates.append({
                "type": "event",
                "confidence": event_confidence,
                "event_types": event_types,
            })

        # Candidate 2: Object intent (if entities detected)
        if primary_entities:
            obj_conf = 0.4
            if attributes.get("color"):
                obj_conf += 0.2  # Color attribute increases confidence
            if attributes.get("vehicle_type"):
                obj_conf += 0.1
            candidates.append({
                "type": "object",
                "confidence": min(obj_conf, 0.8),
                "entities": primary_entities,
            })

        # Candidate 3: Vague violation intent (fallback for fuzzy queries)
        vague_keywords = ["违规", "违章", "违法", "不正常", "异常", "不合规", "不按规定"]
        if any(kw in normalized_query for kw in vague_keywords):
            candidates.append({
                "type": "event",
                "confidence": 0.5,
                "event_types": [],  # No specific event type, search all events
                "vague": True,
            })

        # Sort by confidence descending
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates[:3]  # Keep Top-3

    def _build_rewrites(
        self,
        original_query: str,
        primary_entities: List[str],
        context_entities: List[str],
        relations: List[str],
        directions: List[str],
        motions: List[str],
        event_types: List[str],
        attributes: Dict[str, Any],
    ) -> List[str]:
        """Build retrieval-friendly query variants.

        When use_chinese_clip is True, generates Chinese expansions for CN-CLIP.
        Otherwise, falls back to legacy English rewrites for OpenCLIP.
        """
        if self.use_chinese_clip:
            return self._build_chinese_rewrites(
                original_query=original_query,
                primary_entities=primary_entities,
                context_entities=context_entities,
                event_types=event_types,
                attributes=attributes,
            )

        # Legacy English rewrite path (for OpenCLIP backward compatibility)
        rewrites: List[str] = []
        if original_query:
            rewrites.append(original_query)

        if event_types:
            for event_type in event_types:
                if event_type not in rewrites:
                    rewrites.append(event_type)
            return rewrites

        for entity in primary_entities:
            text = self._display_entity(entity)
            if text not in rewrites:
                rewrites.append(text)

        for entity in primary_entities:
            entity_text = self._display_entity(entity)
            for direction in directions:
                phrase = f"{entity_text} {direction.replace('_', ' ')}"
                if phrase not in rewrites:
                    rewrites.append(phrase)
            for motion in motions:
                phrase = f"{entity_text} {motion.replace('_', ' ')}"
                if phrase not in rewrites:
                    rewrites.append(phrase)
            for relation in relations:
                for context in context_entities:
                    phrase = f"{entity_text} {relation.replace('_', ' ')} {self._display_entity(context)}"
                    if phrase not in rewrites:
                        rewrites.append(phrase)

        if "light_state" in attributes and "traffic_light" in context_entities and primary_entities:
            for entity in primary_entities:
                phrase = f"{self._display_entity(entity)} at {attributes['light_state']} light"
                if phrase not in rewrites:
                    rewrites.append(phrase)

        if "color" in attributes and primary_entities:
            for entity in primary_entities:
                phrase = f"{attributes['color']} {self._display_entity(entity)}"
                if phrase not in rewrites:
                    rewrites.append(phrase)

        return rewrites

    def _build_chinese_rewrites(
        self,
        original_query: str,
        primary_entities: List[str],
        context_entities: List[str],
        event_types: List[str],
        attributes: Dict[str, Any],
    ) -> List[str]:
        """Generate Chinese query expansions for CN-CLIP (Layer 5).

        Instead of translating to English, produces Chinese variants
        that preserve the original semantic intent.
        """
        rewrites: List[str] = []
        if original_query:
            rewrites.append(original_query)

        # Event types → Chinese display names
        for event_type in event_types:
            cn_name = EVENT_CN_DISPLAY.get(event_type, event_type)
            if cn_name not in rewrites:
                rewrites.append(cn_name)

        # Also add event synonym expansions
        for event_type in event_types:
            for expansion in self.expand_query(original_query):
                if expansion not in rewrites:
                    rewrites.append(expansion)

        # Attribute + entity combinations in Chinese
        color = attributes.get("color", "")
        vehicle_type = attributes.get("vehicle_type", "")
        light_state = attributes.get("light_state", "")

        # Map English attribute values to Chinese for CN-CLIP
        COLOR_CN = {"white": "白色", "red": "红色", "black": "黑色",
                    "blue": "蓝色", "yellow": "黄色"}
        VEHICLE_CN = {"car": "汽车", "truck": "货车", "bus": "公交车",
                       "motorcycle": "摩托车"}
        cn_color = COLOR_CN.get(color, color)

        # Build Chinese entity text
        if vehicle_type:
            cn_vehicle = VEHICLE_CN.get(vehicle_type, _ENTITY_CN_DISPLAY.get(vehicle_type, vehicle_type))
        else:
            # Use primary_entities if no vehicle_type attribute
            cn_entities = [_ENTITY_CN_DISPLAY.get(e, e) for e in primary_entities if e in _ENTITY_CN_DISPLAY]
            cn_vehicle = cn_entities[0] if cn_entities else ""

        # Color + vehicle combinations (in Chinese)
        if cn_color and cn_vehicle:
            combos = [
                f"{cn_color}{cn_vehicle}",
                f"{cn_color}的{cn_vehicle}",
                f"{cn_color}车辆",
                f"{cn_color}车",
            ]
            for combo in combos:
                if combo not in rewrites:
                    rewrites.append(combo)
        elif cn_color:
            if f"{cn_color}车" not in rewrites:
                rewrites.append(f"{cn_color}车")
            if f"{cn_color}车辆" not in rewrites:
                rewrites.append(f"{cn_color}车辆")
        elif cn_vehicle:
            if cn_vehicle not in rewrites:
                rewrites.append(cn_vehicle)

        # Light state context
        if light_state == "red":
            if "红灯" not in rewrites:
                rewrites.append("红灯")

        return rewrites

    def _infer_event_types(
        self,
        primary_entities: List[str],
        context_entities: List[str],
        relations: List[str],
        motions: List[str],
        attributes: Dict[str, Any],
        normalized_query: str,
    ) -> List[str]:
        vehicle_entities = {
            entity
            for entity in primary_entities
            if self.ontology["entities"].get(entity, {}).get("group") == "vehicle"
        }
        context_set = set(context_entities)
        relation_set = set(relations)
        motion_set = set(motions)
        event_types: List[str] = []

        if vehicle_entities and "cross" in relation_set and "stop_line" in context_set:
            event_types.append("vehicle_crosses_line")

        if (
            vehicle_entities
            and "cross" in relation_set
            and "stop_line" in context_set
            and attributes.get("light_state") == "red"
            and "traffic_light" in context_set
        ):
            event_types.append("red_light_violation")

        if "red light violation" in normalized_query or "red_light_violation" in normalized_query or "闯红灯" in normalized_query or "冲红灯" in normalized_query or "红灯违规" in normalized_query or "红灯违章" in normalized_query or "红灯违法" in normalized_query:
            event_types.append("red_light_violation")

        if "cross line" in normalized_query or "vehicle_crosses_line" in normalized_query:
            event_types.append("vehicle_crosses_line")

        if motion_set and "intersection" in context_set:
            # Reserved extension hook for future rule-based intersection events.
            pass

        return self._dedupe(event_types)

    def _detect_named_values(self, normalized_query: str, mapping: Dict[str, Iterable[str]]) -> List[str]:
        matches: List[str] = []
        for canonical, aliases in mapping.items():
            if any(self.normalize_label_query(alias) in normalized_query for alias in aliases):
                matches.append(canonical)
        return matches

    def _detect_attributes(self, normalized_query: str) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if "red light" in normalized_query or ("\u7ea2\u706f" in normalized_query and "\u7ea2\u7eff\u706f" not in normalized_query):
            attributes["light_state"] = "red"
        elif "yellow light" in normalized_query or ("\u9ec4\u706f" in normalized_query and "\u7ea2\u7eff\u706f" not in normalized_query):
            attributes["light_state"] = "yellow"
        elif "green light" in normalized_query or ("\u7eff\u706f" in normalized_query and "\u7ea2\u7eff\u706f" not in normalized_query):
            attributes["light_state"] = "green"

        if any(token in normalized_query for token in ("\u957f\u65f6\u95f4", "long time", "for a long time")):
            attributes["duration"] = "long"

        if any(token in normalized_query for token in ("white", "\u767d\u8272")):
            attributes["color"] = "white"
        elif any(token in normalized_query for token in ("black", "\u9ed1\u8272")):
            attributes["color"] = "black"

        return attributes

    def _display_entity(self, entity: str) -> str:
        if self.use_chinese_clip:
            return _ENTITY_CN_DISPLAY.get(entity, entity.replace("_", " "))
        return entity.replace("_", " ")

    def _find_all(self, haystack: str, needle: str) -> List[int]:
        if not needle:
            return []
        positions: List[int] = []
        start = 0
        while True:
            index = haystack.find(needle, start)
            if index == -1:
                break
            positions.append(index)
            start = index + len(needle)
        return positions

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        output: List[str] = []
        for value in values:
            if value not in output:
                output.append(value)
        return output

    def _is_english_query(self, query: str) -> bool:
        cleaned = re.sub(r"[\s_\-]+", "", query)
        return bool(cleaned) and all(ord(char) < 128 for char in cleaned)
