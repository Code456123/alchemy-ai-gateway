"""Lightweight metadata extractor for cache verification.

Extracts intent, entities, and relationships from a query using rule-based NLP.
No LLM calls — runs entirely on CPU in <1ms.
"""

from __future__ import annotations

import re

from loguru import logger

from backend.app.models.cache import CacheMetadata

# Intent patterns: maps regex → normalized intent label.
_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:how\s+much|cost|price|pricing|expensive)\b", re.I), "price_inquiry"),
    (re.compile(r"\b(?:how\s+to|steps?\s+to|guide|tutorial|explain\s+how)\b", re.I), "how_to"),
    (re.compile(r"\b(?:what\s+is|define|meaning|definition)\b", re.I), "definition"),
    (re.compile(r"\b(?:compare|difference|vs\.?|versus)\b", re.I), "comparison"),
    (re.compile(r"\b(?:list|enumerate|give\s+me|show\s+me)\b", re.I), "listing"),
    (re.compile(r"\b(?:why|reason|cause|because)\b", re.I), "reasoning"),
    (re.compile(r"\b(?:write|create|generate|make|build|implement)\b", re.I), "creation"),
    (re.compile(r"\b(?:fix|debug|error|bug|issue|problem|solve)\b", re.I), "troubleshooting"),
    (re.compile(r"\b(?:summarize|summary|tldr|brief)\b", re.I), "summarization"),
    (re.compile(r"\b(?:translate|convert|transform)\b", re.I), "transformation"),
    (re.compile(r"\b(?:recommend|suggest|best|top|should\s+i)\b", re.I), "recommendation"),
    (re.compile(r"\b(?:maintenance|maintain|upkeep|service)\b", re.I), "maintenance"),
)

# Entity extraction: noun-phrase-like tokens after filtering.
_ENTITY_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "and",
        "or",
        "but",
        "not",
        "no",
        "so",
        "if",
        "then",
        "else",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "why",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "about",
        "up",
        "out",
        "off",
        "over",
        "under",
        "between",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "very",
        "just",
        "also",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
        "much",
        "many",
        "few",
        "too",
    }
)

_WORD_RE = re.compile(r"[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*)?", re.I)

# Relationship verbs that connect entities.
_RELATION_VERBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:has|have|having)\b", re.I), "HAS"),
    (re.compile(r"\b(?:is|are|was|were)\b", re.I), "IS"),
    (re.compile(r"\b(?:uses?|using|utilizes?)\b", re.I), "USES"),
    (re.compile(r"\b(?:requires?|needs?|requiring|needing)\b", re.I), "REQUIRES"),
    (re.compile(r"\b(?:contains?|includes?|involving)\b", re.I), "CONTAINS"),
    (re.compile(r"\b(?:creates?|produces?|generates?|builds?)\b", re.I), "CREATES"),
    (re.compile(r"\b(?:runs?|executes?|performs?)\b", re.I), "RUNS"),
    (re.compile(r"\b(?:connects?\s+to|links?\s+to|relates?\s+to)\b", re.I), "CONNECTS_TO"),
)


class MetadataExtractor:
    """Extracts intent, entities, and relationships from a query."""

    def extract(self, query: str) -> CacheMetadata:
        """Extract structured metadata from a raw query.

        Args:
            query: The user's query text.

        Returns:
            A :class:`CacheMetadata` with intent, entities, and relationships.
        """
        intent = self._extract_intent(query)
        entities = self._extract_entities(query)
        relationships = self._extract_relationships(query, entities)

        logger.trace(
            "Metadata extracted: intent={}, entities={}, relationships={}",
            intent,
            entities,
            relationships,
        )
        return CacheMetadata(
            intent=intent,
            entities=entities,
            relationships=relationships,
        )

    def _extract_intent(self, query: str) -> str:
        """Match the query against intent patterns, returning the first match."""
        for pattern, label in _INTENT_PATTERNS:
            if pattern.search(query):
                return label
        return "general"

    def _extract_entities(self, query: str) -> tuple[str, ...]:
        """Extract significant tokens as entities.

        Uses word-level filtering: removes stop words and short words,
        then returns unique remaining tokens in order.
        """
        words = query.lower().split()
        entities: list[str] = []
        for word in words:
            cleaned = re.sub(r"[^a-z0-9]", "", word)
            if (
                cleaned
                and cleaned not in _ENTITY_STOP_WORDS
                and len(cleaned) > 2
                and cleaned not in entities
            ):
                entities.append(cleaned)
        return tuple(entities)

    def _extract_relationships(self, query: str, entities: tuple[str, ...]) -> tuple[str, ...]:
        """Extract entity-relation-entity triples from the query.

        Format: "entity|RELATION|target" where the relation is a
        normalized verb detected between entity mentions.
        """
        if len(entities) < 2:
            return ()

        relations: list[str] = []
        detected_verbs: list[str] = []
        for pattern, label in _RELATION_VERBS:
            if pattern.search(query):
                detected_verbs.append(label)

        if not detected_verbs:
            detected_verbs = ["RELATED_TO"]

        primary_verb = detected_verbs[0]
        # Connect the first entity to subsequent ones via the primary verb.
        subject = entities[0]
        for obj in entities[1:]:
            triple = f"{subject}|{primary_verb}|{obj}"
            if triple not in relations:
                relations.append(triple)

        return tuple(relations)
