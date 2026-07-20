from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from core.schemas import QueryIntent


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
    "vehicle_crosses_line": ("vehicle_crosses_line", "cross line", "line crossing"),
    "red_light_violation": ("red_light_violation", "red light violation"),
}


class QueryRewriteService:
    """Parse deterministic multi-entity traffic intents and derive retrieval-friendly rewrites."""

    def __init__(self) -> None:
        self.ontology = TRAFFIC_ONTOLOGY

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

    def normalize_label_query(self, query: str) -> str:
        """Normalize a free-form traffic query fragment for deterministic matching."""
        return re.sub(r"\s+", " ", query.strip().casefold())

    def parse_query_intent(self, query: str) -> QueryIntent:
        """Parse a traffic short query into a multi-entity intent model."""
        normalized_query = self.normalize_label_query(query)
        if not normalized_query:
            return QueryIntent()

        mentions = self._detect_entity_mentions(normalized_query)
        primary_entities, context_entities = self._split_entities(mentions)
        relations = self._detect_named_values(normalized_query, self.ontology["relations"])
        directions = self._detect_named_values(normalized_query, self.ontology["directions"])
        motions = self._detect_named_values(normalized_query, self.ontology["motions"])
        event_types = self._detect_named_values(normalized_query, _EVENT_ALIASES)
        attributes = self._detect_attributes(normalized_query)

        if "light_state" in attributes and "traffic_light" not in context_entities and "traffic_light" not in primary_entities:
            context_entities = ["traffic_light", *context_entities]

        inferred_event_types = self._infer_event_types(
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            motions=motions,
            attributes=attributes,
            normalized_query=normalized_query,
        )
        event_types = self._dedupe([*event_types, *inferred_event_types])

        query_type = self._classify_query_type(
            primary_entities=primary_entities,
            context_entities=context_entities,
            relations=relations,
            directions=directions,
            motions=motions,
            event_types=event_types,
            attributes=attributes,
        )

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
            attributes=attributes,
            rewritten_queries=rewritten_queries,
            normalized_query=normalized_query,
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

        if "red light violation" in normalized_query or "red_light_violation" in normalized_query or "\u95ef\u7ea2\u706f" in normalized_query:
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
