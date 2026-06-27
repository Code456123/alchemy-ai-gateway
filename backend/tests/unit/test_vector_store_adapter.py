"""Unit tests for vector store adapters and the create_vector_adapter factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.context.memory.vector_store_adapter import (
    LocalVectorStoreAdapter,
    PineconeAdapter,
    create_vector_adapter,
)

# ── Factory Tests ──────────────────────────


def test_factory_returns_local_when_no_pinecone_key() -> None:
    adapter = create_vector_adapter(
        pinecone_api_key="",
        pinecone_index_name="",
        dimension=128,
    )
    assert isinstance(adapter, LocalVectorStoreAdapter)


def test_factory_returns_pinecone_when_configured() -> None:
    with patch("pinecone.Pinecone") as mock_pc_cls:
        mock_pc_cls.return_value.Index.return_value = MagicMock()
        adapter = create_vector_adapter(
            pinecone_api_key="test-key",
            pinecone_index_name="test-index",
            pinecone_namespace="test-ns",
            dimension=512,
        )
    assert isinstance(adapter, PineconeAdapter)


# ── Local Adapter Tests ──────────────────────────


class TestLocalAdapter:
    @pytest.fixture
    def adapter(self) -> LocalVectorStoreAdapter:
        return LocalVectorStoreAdapter(dimension=4)

    def test_upsert_and_query(self, adapter: LocalVectorStoreAdapter) -> None:
        adapter.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1", "text": "hello"})
        results = adapter.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].similarity > 0.9

    def test_query_with_filter(self, adapter: LocalVectorStoreAdapter) -> None:
        adapter.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"})
        adapter.upsert("c2", [0.9, 0.1, 0.0, 0.0], {"session_id": "s2"})
        results = adapter.query([1.0, 0.0, 0.0, 0.0], top_k=5, filter_dict={"session_id": "s2"})
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_delete(self, adapter: LocalVectorStoreAdapter) -> None:
        adapter.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"})
        adapter.delete("c1")
        results = adapter.query([1.0, 0.0, 0.0, 0.0])
        assert len(results) == 0

    def test_delete_by_session(self, adapter: LocalVectorStoreAdapter) -> None:
        adapter.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"})
        adapter.upsert("c2", [0.0, 1.0, 0.0, 0.0], {"session_id": "s1"})
        adapter.upsert("c3", [0.0, 0.0, 1.0, 0.0], {"session_id": "s2"})
        adapter.delete_by_session("s1")
        results = adapter.query([0.0, 0.0, 1.0, 0.0])
        assert len(results) == 1
        assert results[0].chunk_id == "c3"

    def test_clear(self, adapter: LocalVectorStoreAdapter) -> None:
        adapter.upsert("c1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"})
        adapter.clear()
        assert adapter.query([1.0, 0.0, 0.0, 0.0]) == []

    def test_empty_query_returns_empty(self, adapter: LocalVectorStoreAdapter) -> None:
        assert adapter.query([1.0, 0.0, 0.0, 0.0]) == []


# ── Pinecone Adapter Tests (mocked SDK) ──────────────────────────


class TestPineconeAdapter:
    @pytest.fixture
    def mock_index(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def adapter(self, mock_index: MagicMock) -> PineconeAdapter:
        with patch("pinecone.Pinecone") as mock_pc_cls:
            mock_pc_cls.return_value.Index.return_value = mock_index
            return PineconeAdapter(
                api_key="test-key",
                index_name="test-index",
                namespace="test-ns",
            )

    def test_upsert_calls_pinecone(self, adapter: PineconeAdapter, mock_index: MagicMock) -> None:
        adapter.upsert("c1", [0.1, 0.2], {"text": "hello", "session_id": "s1"})
        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args
        vectors = call_kwargs.kwargs.get("vectors") or call_kwargs[1].get("vectors")
        assert vectors[0][0] == "c1"

    def test_upsert_truncates_long_text(
        self, adapter: PineconeAdapter, mock_index: MagicMock
    ) -> None:
        long_text = "x" * 2000
        adapter.upsert("c1", [0.1], {"text": long_text})
        call_kwargs = mock_index.upsert.call_args
        vectors = call_kwargs.kwargs.get("vectors") or call_kwargs[1].get("vectors")
        stored_meta = vectors[0][2]
        assert len(stored_meta["text"]) == 1000

    def test_query_returns_results(self, adapter: PineconeAdapter, mock_index: MagicMock) -> None:
        mock_match = MagicMock()
        mock_match.id = "c1"
        mock_match.score = 0.95
        mock_match.metadata = {"text": "hello", "session_id": "s1"}
        mock_index.query.return_value.matches = [mock_match]

        results = adapter.query([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].similarity == 0.95
        assert results[0].metadata["text"] == "hello"

    def test_query_with_filter(self, adapter: PineconeAdapter, mock_index: MagicMock) -> None:
        mock_index.query.return_value.matches = []
        adapter.query([0.1], top_k=3, filter_dict={"session_id": "s1"})
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["filter"] == {"session_id": "s1"}

    def test_query_empty_returns_empty(
        self, adapter: PineconeAdapter, mock_index: MagicMock
    ) -> None:
        mock_index.query.return_value.matches = []
        results = adapter.query([0.1])
        assert results == []

    def test_delete_calls_pinecone(self, adapter: PineconeAdapter, mock_index: MagicMock) -> None:
        adapter.delete("c1")
        mock_index.delete.assert_called_once_with(ids=["c1"], namespace="test-ns")

    def test_delete_by_session_calls_pinecone(
        self, adapter: PineconeAdapter, mock_index: MagicMock
    ) -> None:
        adapter.delete_by_session("s1")
        mock_index.delete.assert_called_once_with(filter={"session_id": "s1"}, namespace="test-ns")

    def test_clear_calls_pinecone(self, adapter: PineconeAdapter, mock_index: MagicMock) -> None:
        adapter.clear()
        mock_index.delete.assert_called_once_with(delete_all=True, namespace="test-ns")

    def test_namespace_is_used_in_all_calls(
        self, adapter: PineconeAdapter, mock_index: MagicMock
    ) -> None:
        adapter.upsert("c1", [0.1], {"text": "x"})
        assert mock_index.upsert.call_args.kwargs["namespace"] == "test-ns"

        mock_index.query.return_value.matches = []
        adapter.query([0.1])
        assert mock_index.query.call_args.kwargs["namespace"] == "test-ns"
