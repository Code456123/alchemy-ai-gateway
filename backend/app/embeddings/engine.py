"""Lightweight embedding engine for semantic cache.

Produces normalized bag-of-words TF vectors as a zero-dependency stand-in for
sentence-transformers. The interface is designed so that swapping in a real
SentenceTransformer model requires changing only the ``_encode`` method.

Performance: <1ms per query embedding on any hardware.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from loguru import logger

# Dimensionality of the BoW vectors — chosen to balance FAISS index size
# and collision risk. A real model would output 384 (MiniLM) or 768 (BERT).
_VOCAB_DIM = 512

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common English stop words filtered out to sharpen similarity.
_STOP_WORDS: frozenset[str] = frozenset(
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
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "so",
        "yet",
        "both",
        "either",
        "neither",
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
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
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
        "what",
        "which",
        "who",
        "whom",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, and remove stop words."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def _hash_token(token: str, dim: int) -> int:
    """Deterministic hash of a token into [0, dim)."""
    h = 5381
    for ch in token:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h % dim


class EmbeddingEngine:
    """Produces fixed-dimension normalized vectors from text."""

    def __init__(self, dim: int = _VOCAB_DIM) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        """The dimensionality of output vectors."""
        return self._dim

    def encode(self, text: str) -> list[float]:
        """Encode a query into a normalized embedding vector.

        Args:
            text: Raw query string.

        Returns:
            A list of floats of length ``dimension``, L2-normalized.
        """
        tokens = _tokenize(text)
        vec = [0.0] * self._dim

        counts = Counter(tokens)
        for token, count in counts.items():
            idx = _hash_token(token, self._dim)
            vec[idx] += float(count)

        # L2 normalize.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        logger.trace("Encoded {} tokens → dim-{} vector", len(tokens), self._dim)
        return vec

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Both vectors are assumed to be L2-normalized, so this is just
        the dot product.
        """
        return sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
