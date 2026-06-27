"""Unit tests for SemanticChunker."""

import pytest

from backend.app.context.chunking import SemanticChunker
from backend.app.context.models import ChunkType, Speaker


class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(min_tokens=50, max_tokens=100)

    def test_short_message_single_chunk(self):
        chunks = self.chunker.chunk_message(
            text="Hello world",
            session_id="s1",
            speaker=Speaker.USER,
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].session_id == "s1"
        assert chunks[0].speaker == Speaker.USER

    def test_empty_message_no_chunks(self):
        chunks = self.chunker.chunk_message(
            text="",
            session_id="s1",
            speaker=Speaker.USER,
        )
        assert len(chunks) == 0

    def test_long_message_splits(self):
        text = ". ".join(f"This is sentence number {i}" for i in range(50))
        chunks = self.chunker.chunk_message(
            text=text,
            session_id="s1",
            speaker=Speaker.USER,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.session_id == "s1"

    def test_chunk_metadata(self):
        chunks = self.chunker.chunk_message(
            text="Explain photosynthesis in detail",
            session_id="s1",
            speaker=Speaker.USER,
            chunk_type=ChunkType.USER_QUERY,
            model_used="gpt4o",
            topic="science",
        )
        assert len(chunks) == 1
        assert chunks[0].topic == "science"
        assert chunks[0].model_used == "gpt4o"
        assert chunks[0].chunk_type == ChunkType.USER_QUERY

    def test_chunk_conversation(self):
        messages = [
            (Speaker.USER, "Hello", ""),
            (Speaker.ASSISTANT, "Hi there!", "gpt4o"),
            (Speaker.USER, "How are you?", ""),
        ]
        chunks = self.chunker.chunk_conversation(messages, "s1")
        assert len(chunks) == 3
        assert chunks[0].speaker == Speaker.USER
        assert chunks[1].speaker == Speaker.ASSISTANT
