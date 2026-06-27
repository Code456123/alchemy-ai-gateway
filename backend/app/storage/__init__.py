"""Storage layer — SQLite persistence, FAISS index management, CRUD operations."""

from __future__ import annotations

from backend.app.storage.vector_store import SearchResult, VectorStore

__all__ = ["SearchResult", "VectorStore"]
