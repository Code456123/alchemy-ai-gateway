"""Unit tests for WorkingMemory."""

import pytest

from backend.app.context.memory.working_memory import WorkingMemory
from backend.app.context.models import SemanticChunk, Speaker


class TestWorkingMemory:
    def setup_method(self):
        self.wm = WorkingMemory(max_chunks=5)

    def test_add_and_retrieve(self):
        chunk = SemanticChunk(session_id="s1", text="test", speaker=Speaker.USER)
        self.wm.add(chunk)
        assert self.wm.size == 1
        assert self.wm.chunks[0].text == "test"

    def test_max_size_eviction(self):
        for i in range(10):
            self.wm.add(SemanticChunk(session_id="s1", text=f"chunk-{i}", speaker=Speaker.USER))
        assert self.wm.size == 5
        assert self.wm.chunks[0].text == "chunk-5"

    def test_get_recent(self):
        for i in range(5):
            self.wm.add(SemanticChunk(session_id="s1", text=f"chunk-{i}", speaker=Speaker.USER))
        recent = self.wm.get_recent(2)
        assert len(recent) == 2
        assert recent[0].text == "chunk-3"
        assert recent[1].text == "chunk-4"

    def test_search_by_topic(self):
        self.wm.add(SemanticChunk(session_id="s1", text="python code", topic="coding", speaker=Speaker.USER))
        self.wm.add(SemanticChunk(session_id="s1", text="hello world", topic="greeting", speaker=Speaker.USER))
        results = self.wm.search_by_topic("coding")
        assert len(results) == 1
        assert results[0].topic == "coding"

    def test_total_tokens(self):
        self.wm.add(SemanticChunk(session_id="s1", text="test", token_count=10, speaker=Speaker.USER))
        self.wm.add(SemanticChunk(session_id="s1", text="test2", token_count=20, speaker=Speaker.USER))
        assert self.wm.total_tokens() == 30

    def test_clear(self):
        self.wm.add(SemanticChunk(session_id="s1", text="test", speaker=Speaker.USER))
        self.wm.clear()
        assert self.wm.size == 0
