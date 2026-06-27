"""Unit tests for the stateful pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.constants.enums import (
    FastRequestCategory,
    RoutingAction,
    SecurityStatus,
    TaskType,
    ThreatType,
)
from backend.app.constants.models import ModelID
from backend.app.models.analysis import FastDetectorResult, PromptAnalysis, SecurityResult
from backend.app.models.request import PromptRequest
from backend.app.pipeline.checkpoint_manager import (
    CheckpointBackend,
    CheckpointManager,
    InMemoryCheckpointBackend,
)
from backend.app.pipeline.event_dispatcher import EventDispatcher, PipelineEvent
from backend.app.pipeline.exceptions import (
    CheckpointError,
    PipelineError,
    PipelineTerminated,
    StageExecutionError,
    StageTimeoutError,
)
from backend.app.pipeline.execution_trace import ExecutionTrace
from backend.app.pipeline.pipeline_context import PipelineContext, PipelineStatus
from backend.app.pipeline.retry_manager import (
    DEFAULT_RETRY_CONFIGS,
    RetryConfig,
    RetryManager,
    RetryStrategy,
)
from backend.app.pipeline.stage_executor import StageExecutor
from backend.app.pipeline.stage_status import StageName, StageRecord, StageStatus


# ━━ Fixtures ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def dispatcher() -> EventDispatcher:
    return EventDispatcher()


@pytest.fixture
def retry_manager() -> RetryManager:
    return RetryManager()


@pytest.fixture
def checkpoint_backend() -> InMemoryCheckpointBackend:
    return InMemoryCheckpointBackend()


@pytest.fixture
def checkpoint_manager(
    checkpoint_backend: InMemoryCheckpointBackend, dispatcher: EventDispatcher
) -> CheckpointManager:
    return CheckpointManager(backend=checkpoint_backend, dispatcher=dispatcher)


@pytest.fixture
def stage_executor(
    retry_manager: RetryManager,
    checkpoint_manager: CheckpointManager,
    dispatcher: EventDispatcher,
) -> StageExecutor:
    return StageExecutor(retry_manager, checkpoint_manager, dispatcher)


@pytest.fixture
def context() -> PipelineContext:
    return PipelineContext(
        request_id="test-123",
        user_query="What is Python?",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StageStatus / StageRecord
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStageStatus:
    def test_all_status_values_exist(self) -> None:
        assert StageStatus.PENDING == "PENDING"
        assert StageStatus.RUNNING == "RUNNING"
        assert StageStatus.COMPLETED == "COMPLETED"
        assert StageStatus.FAILED == "FAILED"
        assert StageStatus.SKIPPED == "SKIPPED"

    def test_stage_record_creation(self) -> None:
        record = StageRecord(
            name=StageName.FAST_DETECTOR,
            status=StageStatus.COMPLETED,
            latency_ms=1.5,
        )
        assert record.name == StageName.FAST_DETECTOR
        assert record.status == StageStatus.COMPLETED
        assert record.latency_ms == 1.5
        assert record.error is None
        assert record.retry_count == 0
        assert record.checkpoint_time is None

    def test_stage_record_with_error(self) -> None:
        record = StageRecord(
            name=StageName.SECURITY,
            status=StageStatus.FAILED,
            latency_ms=3.0,
            retry_count=2,
            error="connection timeout",
        )
        assert record.error == "connection timeout"
        assert record.retry_count == 2

    def test_stage_record_frozen(self) -> None:
        record = StageRecord(name=StageName.BUDGET, status=StageStatus.COMPLETED)
        with pytest.raises(Exception):
            record.status = StageStatus.FAILED  # type: ignore[misc]

    def test_all_stage_names_exist(self) -> None:
        expected = {
            "fast_detector",
            "security",
            "task_analyzer",
            "decision_engine",
            "budget",
            "semantic_cache",
            "context_manager",
            "response_generation",
            "cache_store",
        }
        assert {s.value for s in StageName} == expected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ExecutionTrace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExecutionTrace:
    def test_add_and_query(self) -> None:
        trace = ExecutionTrace()
        trace.add(
            StageRecord(
                name=StageName.FAST_DETECTOR,
                status=StageStatus.COMPLETED,
                latency_ms=1.0,
            )
        )
        trace.add(StageRecord(name=StageName.SECURITY, status=StageStatus.SKIPPED))
        trace.add(
            StageRecord(
                name=StageName.TASK_ANALYZER,
                status=StageStatus.FAILED,
                error="boom",
            )
        )

        assert trace.completed_stages == [StageName.FAST_DETECTOR]
        assert trace.skipped_stages == [StageName.SECURITY]
        assert trace.failed_stages == [StageName.TASK_ANALYZER]
        assert trace.total_latency_ms == 1.0

    def test_get_by_name(self) -> None:
        trace = ExecutionTrace()
        trace.add(
            StageRecord(
                name=StageName.BUDGET, status=StageStatus.COMPLETED, latency_ms=0.5
            )
        )
        assert trace.get(StageName.BUDGET) is not None
        assert trace.get(StageName.SECURITY) is None

    def test_summary_format(self) -> None:
        trace = ExecutionTrace()
        trace.add(
            StageRecord(
                name=StageName.FAST_DETECTOR,
                status=StageStatus.COMPLETED,
                latency_ms=2.0,
            )
        )
        summary = trace.summary()
        assert len(summary) == 1
        assert summary[0]["stage"] == "fast_detector"
        assert summary[0]["status"] == "COMPLETED"
        assert summary[0]["latency_ms"] == 2.0
        assert summary[0]["retry_count"] == 0
        assert summary[0]["error"] is None

    def test_empty_trace(self) -> None:
        trace = ExecutionTrace()
        assert trace.completed_stages == []
        assert trace.failed_stages == []
        assert trace.skipped_stages == []
        assert trace.total_latency_ms == 0.0
        assert trace.summary() == []

    def test_total_latency_sums_all(self) -> None:
        trace = ExecutionTrace()
        trace.add(
            StageRecord(
                name=StageName.FAST_DETECTOR,
                status=StageStatus.COMPLETED,
                latency_ms=1.5,
            )
        )
        trace.add(
            StageRecord(
                name=StageName.SECURITY,
                status=StageStatus.COMPLETED,
                latency_ms=2.5,
            )
        )
        trace.add(
            StageRecord(
                name=StageName.TASK_ANALYZER,
                status=StageStatus.SKIPPED,
                latency_ms=0.0,
            )
        )
        assert trace.total_latency_ms == 4.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EventDispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventDispatcher:
    def test_emit_to_subscriber(self, dispatcher: EventDispatcher) -> None:
        received: list[tuple[PipelineEvent, dict]] = []
        dispatcher.subscribe(
            PipelineEvent.STAGE_COMPLETED, lambda e, d: received.append((e, d))
        )
        dispatcher.emit(PipelineEvent.STAGE_COMPLETED, {"stage": "test"})
        assert len(received) == 1
        assert received[0][0] == PipelineEvent.STAGE_COMPLETED
        assert received[0][1]["stage"] == "test"

    def test_multiple_subscribers(self, dispatcher: EventDispatcher) -> None:
        a: list[str] = []
        b: list[str] = []
        dispatcher.subscribe(PipelineEvent.PIPELINE_COMPLETED, lambda e, d: a.append("a"))
        dispatcher.subscribe(PipelineEvent.PIPELINE_COMPLETED, lambda e, d: b.append("b"))
        dispatcher.emit(PipelineEvent.PIPELINE_COMPLETED, {})
        assert a == ["a"]
        assert b == ["b"]

    def test_handler_error_does_not_propagate(self, dispatcher: EventDispatcher) -> None:
        good: list[str] = []

        def bad_handler(e, d):
            raise RuntimeError("handler error")

        dispatcher.subscribe(PipelineEvent.STAGE_COMPLETED, bad_handler)
        dispatcher.subscribe(PipelineEvent.STAGE_COMPLETED, lambda e, d: good.append("ok"))
        dispatcher.emit(PipelineEvent.STAGE_COMPLETED, {})
        assert good == ["ok"]

    def test_no_subscribers_does_not_fail(self, dispatcher: EventDispatcher) -> None:
        dispatcher.emit(PipelineEvent.PIPELINE_FAILED, {"error": "test"})

    def test_emit_with_none_data(self, dispatcher: EventDispatcher) -> None:
        received: list[dict] = []
        dispatcher.subscribe(
            PipelineEvent.CACHE_MISS, lambda e, d: received.append(d)
        )
        dispatcher.emit(PipelineEvent.CACHE_MISS, None)
        assert received == [{}]

    def test_all_events_exist(self) -> None:
        expected = {
            "PIPELINE_STARTED",
            "STAGE_STARTED",
            "STAGE_COMPLETED",
            "STAGE_FAILED",
            "CHECKPOINT_CREATED",
            "CHECKPOINT_RESTORED",
            "RETRY_ATTEMPT",
            "FAST_RESPONSE",
            "SECURITY_BLOCKED",
            "CACHE_HIT",
            "CACHE_MISS",
            "CONTEXT_READY",
            "RESPONSE_SUCCESS",
            "RESPONSE_FAILED",
            "PIPELINE_COMPLETED",
            "PIPELINE_FAILED",
        }
        assert {e.value for e in PipelineEvent} == expected

    def test_unrelated_event_not_triggered(self, dispatcher: EventDispatcher) -> None:
        received: list[str] = []
        dispatcher.subscribe(
            PipelineEvent.CACHE_HIT, lambda e, d: received.append("hit")
        )
        dispatcher.emit(PipelineEvent.CACHE_MISS, {})
        assert received == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RetryManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRetryManager:
    def test_respects_max_retries(self) -> None:
        mgr = RetryManager({StageName.SECURITY: RetryConfig(max_retries=2)})
        assert mgr.can_retry(StageName.SECURITY)
        mgr.record_attempt(StageName.SECURITY, "err1")
        assert mgr.can_retry(StageName.SECURITY)
        mgr.record_attempt(StageName.SECURITY, "err2")
        assert not mgr.can_retry(StageName.SECURITY)

    def test_zero_retries_means_no_retry(self) -> None:
        mgr = RetryManager({StageName.FAST_DETECTOR: RetryConfig(max_retries=0)})
        assert not mgr.can_retry(StageName.FAST_DETECTOR)

    def test_reset_single_stage(self) -> None:
        mgr = RetryManager({StageName.SECURITY: RetryConfig(max_retries=1)})
        mgr.record_attempt(StageName.SECURITY, "err")
        assert not mgr.can_retry(StageName.SECURITY)
        mgr.reset(StageName.SECURITY)
        assert mgr.can_retry(StageName.SECURITY)

    def test_reset_all(self) -> None:
        mgr = RetryManager({
            StageName.SECURITY: RetryConfig(max_retries=1),
            StageName.BUDGET: RetryConfig(max_retries=1),
        })
        mgr.record_attempt(StageName.SECURITY, "err")
        mgr.record_attempt(StageName.BUDGET, "err")
        mgr.reset_all()
        assert mgr.can_retry(StageName.SECURITY)
        assert mgr.can_retry(StageName.BUDGET)

    def test_get_attempt_count(self) -> None:
        mgr = RetryManager({StageName.SECURITY: RetryConfig(max_retries=5)})
        assert mgr.get_attempt_count(StageName.SECURITY) == 0
        mgr.record_attempt(StageName.SECURITY, "err1")
        assert mgr.get_attempt_count(StageName.SECURITY) == 1
        mgr.record_attempt(StageName.SECURITY, "err2")
        assert mgr.get_attempt_count(StageName.SECURITY) == 2

    def test_get_config_returns_fallback_for_empty_configs(self) -> None:
        mgr = RetryManager({})
        config = mgr.get_config(StageName.BUDGET)
        assert isinstance(config, RetryConfig)

    def test_fixed_strategy_constant_delay(self) -> None:
        mgr = RetryManager(
            {StageName.SECURITY: RetryConfig(max_retries=3, delay_seconds=0, strategy=RetryStrategy.FIXED)}
        )
        mgr.record_attempt(StageName.SECURITY, "err")
        mgr.wait_before_retry(StageName.SECURITY)

    def test_exponential_strategy(self) -> None:
        mgr = RetryManager(
            {StageName.RESPONSE_GENERATION: RetryConfig(max_retries=3, delay_seconds=0, strategy=RetryStrategy.EXPONENTIAL)}
        )
        mgr.record_attempt(StageName.RESPONSE_GENERATION, "err")
        mgr.wait_before_retry(StageName.RESPONSE_GENERATION)

    def test_default_configs_cover_all_stages(self) -> None:
        for stage in StageName:
            assert stage in DEFAULT_RETRY_CONFIGS

    def test_record_attempt_returns_count(self) -> None:
        mgr = RetryManager({StageName.SECURITY: RetryConfig(max_retries=5)})
        assert mgr.record_attempt(StageName.SECURITY, "e1") == 1
        assert mgr.record_attempt(StageName.SECURITY, "e2") == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PipelineContext
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineContext:
    def test_serialization_roundtrip(self, context: PipelineContext) -> None:
        data = context.serialize()
        restored = PipelineContext.deserialize(data)
        assert restored.request_id == context.request_id
        assert restored.user_query == context.user_query
        assert restored.status == context.status

    def test_serialization_with_stage_results(self) -> None:
        ctx = PipelineContext(request_id="ser-1", user_query="test")
        ctx.fast_detector_result = FastDetectorResult(
            is_fast_path=True,
            category=FastRequestCategory.GREETING,
            reason="greeting",
            canned_response="Hello!",
        )
        ctx.security_result = SecurityResult(
            status=SecurityStatus.CLEAR, reason="ok"
        )
        data = ctx.serialize()
        restored = PipelineContext.deserialize(data)
        assert restored.fast_detector_result is not None
        assert restored.fast_detector_result.is_fast_path is True
        assert restored.security_result is not None
        assert not restored.security_result.is_blocked

    def test_mark_running(self, context: PipelineContext) -> None:
        context.mark_running(StageName.SECURITY)
        assert context.current_stage == StageName.SECURITY
        assert context.status == PipelineStatus.RUNNING

    def test_mark_completed(self, context: PipelineContext) -> None:
        context.mark_completed()
        assert context.status == PipelineStatus.COMPLETED
        assert context.completed_at is not None

    def test_mark_failed(self, context: PipelineContext) -> None:
        context.mark_failed("something broke")
        assert context.status == PipelineStatus.FAILED
        assert context.error == "something broke"
        assert context.metadata["terminal_error"] == "something broke"
        assert context.completed_at is not None

    def test_mark_terminated_early(self, context: PipelineContext) -> None:
        context.mark_terminated_early("cache_hit")
        assert context.status == PipelineStatus.TERMINATED_EARLY
        assert context.metadata["early_termination_reason"] == "cache_hit"
        assert context.completed_at is not None

    def test_elapsed_ms_positive(self, context: PipelineContext) -> None:
        assert context.elapsed_ms >= 0

    def test_elapsed_ms_uses_completed_at_if_set(self) -> None:
        ctx = PipelineContext(request_id="t", user_query="q", started_at=100.0)
        ctx.completed_at = 100.5
        assert ctx.elapsed_ms == 500.0

    def test_completed_stages_property(self) -> None:
        ctx = PipelineContext(request_id="t", user_query="q")
        ctx.execution_trace.add(
            StageRecord(name=StageName.FAST_DETECTOR, status=StageStatus.COMPLETED)
        )
        ctx.execution_trace.add(
            StageRecord(name=StageName.SECURITY, status=StageStatus.SKIPPED)
        )
        assert ctx.completed_stages == [StageName.FAST_DETECTOR]

    def test_default_values(self) -> None:
        ctx = PipelineContext(request_id="t", user_query="q")
        assert ctx.status == PipelineStatus.PENDING
        assert ctx.current_stage is None
        assert ctx.retry_count == 0
        assert ctx.session_id is None
        assert ctx.fast_detector_result is None
        assert ctx.security_result is None
        assert ctx.analysis_result is None
        assert ctx.budget_snapshot is None
        assert ctx.routing_decision is None
        assert ctx.cache_hit is None
        assert ctx.response_text is None
        assert ctx.response_model is None
        assert ctx.error is None
        assert ctx.metadata == {}
        assert ctx.economic_mode is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CheckpointManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckpointManager:
    def test_save_and_load(
        self, checkpoint_manager: CheckpointManager, context: PipelineContext
    ) -> None:
        checkpoint_manager.save_checkpoint(context)
        restored = checkpoint_manager.load_checkpoint(context.request_id)
        assert restored is not None
        assert restored.request_id == context.request_id
        assert restored.user_query == context.user_query

    def test_load_missing_returns_none(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        assert checkpoint_manager.load_checkpoint("nonexistent") is None

    def test_delete(
        self, checkpoint_manager: CheckpointManager, context: PipelineContext
    ) -> None:
        checkpoint_manager.save_checkpoint(context)
        checkpoint_manager.delete_checkpoint(context.request_id)
        assert checkpoint_manager.load_checkpoint(context.request_id) is None

    def test_save_emits_checkpoint_created_event(
        self, dispatcher: EventDispatcher, context: PipelineContext
    ) -> None:
        events: list[PipelineEvent] = []
        dispatcher.subscribe(
            PipelineEvent.CHECKPOINT_CREATED, lambda e, d: events.append(e)
        )
        mgr = CheckpointManager(dispatcher=dispatcher)
        mgr.save_checkpoint(context)
        assert PipelineEvent.CHECKPOINT_CREATED in events

    def test_load_emits_checkpoint_restored_event(
        self, dispatcher: EventDispatcher, context: PipelineContext
    ) -> None:
        events: list[PipelineEvent] = []
        dispatcher.subscribe(
            PipelineEvent.CHECKPOINT_RESTORED, lambda e, d: events.append(e)
        )
        mgr = CheckpointManager(dispatcher=dispatcher)
        mgr.save_checkpoint(context)
        mgr.load_checkpoint(context.request_id)
        assert PipelineEvent.CHECKPOINT_RESTORED in events

    def test_overwrite_checkpoint(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        ctx1 = PipelineContext(request_id="ow-1", user_query="first")
        checkpoint_manager.save_checkpoint(ctx1)
        ctx1.user_query = "second"
        checkpoint_manager.save_checkpoint(ctx1)
        restored = checkpoint_manager.load_checkpoint("ow-1")
        assert restored is not None

    def test_checkpoint_preserves_execution_trace(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        ctx = PipelineContext(request_id="trace-1", user_query="test")
        ctx.execution_trace.add(
            StageRecord(
                name=StageName.FAST_DETECTOR,
                status=StageStatus.COMPLETED,
                latency_ms=1.0,
            )
        )
        ctx.execution_trace.add(
            StageRecord(
                name=StageName.SECURITY,
                status=StageStatus.COMPLETED,
                latency_ms=2.0,
            )
        )
        checkpoint_manager.save_checkpoint(ctx)
        restored = checkpoint_manager.load_checkpoint("trace-1")
        assert restored is not None
        assert len(restored.execution_trace.records) == 2
        assert restored.execution_trace.completed_stages == [
            StageName.FAST_DETECTOR,
            StageName.SECURITY,
        ]

    def test_backend_interface_in_memory(self) -> None:
        backend = InMemoryCheckpointBackend()
        backend.save("k1", '{"data": 1}')
        assert backend.load("k1") == '{"data": 1}'
        backend.delete("k1")
        assert backend.load("k1") is None

    def test_save_failure_raises_checkpoint_error(
        self, dispatcher: EventDispatcher
    ) -> None:
        class FailingBackend(CheckpointBackend):
            def save(self, request_id: str, data: str) -> None:
                raise IOError("disk full")

            def load(self, request_id: str) -> str | None:
                return None

            def delete(self, request_id: str) -> None:
                pass

        mgr = CheckpointManager(backend=FailingBackend(), dispatcher=dispatcher)
        ctx = PipelineContext(request_id="fail", user_query="q")
        with pytest.raises(CheckpointError, match="disk full"):
            mgr.save_checkpoint(ctx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StageExecutor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStageExecutor:
    def test_success(
        self, stage_executor: StageExecutor, context: PipelineContext
    ) -> None:
        def noop(ctx: PipelineContext) -> None:
            ctx.metadata["touched"] = True

        record = stage_executor.execute(StageName.FAST_DETECTOR, noop, context)
        assert record.status == StageStatus.COMPLETED
        assert record.latency_ms >= 0
        assert record.checkpoint_time is not None
        assert context.metadata["touched"] is True
        assert context.current_stage == StageName.FAST_DETECTOR

    def test_failure_no_retry(
        self, dispatcher: EventDispatcher, checkpoint_manager: CheckpointManager
    ) -> None:
        mgr = RetryManager({StageName.FAST_DETECTOR: RetryConfig(max_retries=0)})
        executor = StageExecutor(mgr, checkpoint_manager, dispatcher)
        ctx = PipelineContext(request_id="fail-1", user_query="test")

        def boom(ctx: PipelineContext) -> None:
            raise ValueError("kaboom")

        with pytest.raises(StageExecutionError, match="kaboom"):
            executor.execute(StageName.FAST_DETECTOR, boom, ctx)

        assert len(ctx.execution_trace.failed_stages) == 1

    def test_retries_then_succeeds(
        self, dispatcher: EventDispatcher, checkpoint_manager: CheckpointManager
    ) -> None:
        mgr = RetryManager(
            {StageName.SECURITY: RetryConfig(max_retries=2, delay_seconds=0)}
        )
        executor = StageExecutor(mgr, checkpoint_manager, dispatcher)
        ctx = PipelineContext(request_id="retry-1", user_query="test")

        call_count = 0

        def flaky(ctx: PipelineContext) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient")

        record = executor.execute(StageName.SECURITY, flaky, ctx)
        assert record.status == StageStatus.COMPLETED
        assert call_count == 2
        assert record.retry_count == 1

    def test_exhausts_retries_then_fails(
        self, dispatcher: EventDispatcher, checkpoint_manager: CheckpointManager
    ) -> None:
        mgr = RetryManager(
            {StageName.SEMANTIC_CACHE: RetryConfig(max_retries=2, delay_seconds=0)}
        )
        executor = StageExecutor(mgr, checkpoint_manager, dispatcher)
        ctx = PipelineContext(request_id="exhaust-1", user_query="test")

        def always_fail(ctx: PipelineContext) -> None:
            raise RuntimeError("permanent")

        with pytest.raises(StageExecutionError, match="permanent"):
            executor.execute(StageName.SEMANTIC_CACHE, always_fail, ctx)

    def test_skip(
        self, stage_executor: StageExecutor, context: PipelineContext
    ) -> None:
        record = stage_executor.skip(StageName.CONTEXT_MANAGER, context)
        assert record.status == StageStatus.SKIPPED
        assert StageName.CONTEXT_MANAGER in context.execution_trace.skipped_stages

    def test_emits_stage_started_event(
        self, dispatcher: EventDispatcher, checkpoint_manager: CheckpointManager
    ) -> None:
        events: list[PipelineEvent] = []
        dispatcher.subscribe(
            PipelineEvent.STAGE_STARTED, lambda e, d: events.append(e)
        )
        mgr = RetryManager({StageName.BUDGET: RetryConfig(max_retries=0)})
        executor = StageExecutor(mgr, checkpoint_manager, dispatcher)
        ctx = PipelineContext(request_id="ev-1", user_query="test")
        executor.execute(StageName.BUDGET, lambda c: None, ctx)
        assert PipelineEvent.STAGE_STARTED in events

    def test_emits_stage_completed_event(
        self, dispatcher: EventDispatcher, checkpoint_manager: CheckpointManager
    ) -> None:
        events: list[dict] = []
        dispatcher.subscribe(
            PipelineEvent.STAGE_COMPLETED, lambda e, d: events.append(d)
        )
        mgr = RetryManager({StageName.BUDGET: RetryConfig(max_retries=0)})
        executor = StageExecutor(mgr, checkpoint_manager, dispatcher)
        ctx = PipelineContext(request_id="ev-2", user_query="test")
        executor.execute(StageName.BUDGET, lambda c: None, ctx)
        assert len(events) == 1
        assert events[0]["stage"] == "budget"

    def test_saves_checkpoint_on_success(
        self,
        dispatcher: EventDispatcher,
        checkpoint_backend: InMemoryCheckpointBackend,
    ) -> None:
        cp_mgr = CheckpointManager(backend=checkpoint_backend, dispatcher=dispatcher)
        retry_mgr = RetryManager({StageName.BUDGET: RetryConfig(max_retries=0)})
        executor = StageExecutor(retry_mgr, cp_mgr, dispatcher)
        ctx = PipelineContext(request_id="cp-1", user_query="test")
        executor.execute(StageName.BUDGET, lambda c: None, ctx)
        assert checkpoint_backend.load("cp-1") is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checkpoint Recovery Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCheckpointRecovery:
    def test_resume_skips_completed_stages(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        ctx = PipelineContext(request_id="resume-1", user_query="test")
        ctx.execution_trace.add(
            StageRecord(
                name=StageName.FAST_DETECTOR, status=StageStatus.COMPLETED, latency_ms=1.0
            )
        )
        ctx.execution_trace.add(
            StageRecord(
                name=StageName.SECURITY, status=StageStatus.COMPLETED, latency_ms=1.0
            )
        )
        ctx.current_stage = StageName.TASK_ANALYZER
        checkpoint_manager.save_checkpoint(ctx)

        restored = checkpoint_manager.load_checkpoint("resume-1")
        assert restored is not None
        completed = set(restored.execution_trace.completed_stages)
        assert StageName.FAST_DETECTOR in completed
        assert StageName.SECURITY in completed
        assert StageName.TASK_ANALYZER not in completed

    def test_checkpoint_preserves_stage_results(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        ctx = PipelineContext(request_id="res-1", user_query="test query")
        ctx.security_result = SecurityResult(
            status=SecurityStatus.CLEAR, reason="ok"
        )
        ctx.fast_detector_result = FastDetectorResult(
            is_fast_path=False, reason="non-trivial"
        )
        checkpoint_manager.save_checkpoint(ctx)

        restored = checkpoint_manager.load_checkpoint("res-1")
        assert restored is not None
        assert restored.security_result is not None
        assert not restored.security_result.is_blocked
        assert restored.fast_detector_result is not None
        assert not restored.fast_detector_result.is_fast_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal Stage Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTerminalStageHandling:
    def test_fast_path_skips_stages_after_security(self) -> None:
        ctx = PipelineContext(request_id="term-1", user_query="hello")
        ctx.fast_detector_result = FastDetectorResult(
            is_fast_path=True,
            category=FastRequestCategory.GREETING,
            reason="greeting",
            canned_response="Hello!",
        )
        ctx.security_result = SecurityResult(
            status=SecurityStatus.CLEAR, reason="ok"
        )

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        # After fast_detector + security completed, task_analyzer should be skipped
        assert orch._should_skip(StageName.TASK_ANALYZER, ctx) is True
        assert orch._should_skip(StageName.DECISION_ENGINE, ctx) is True
        assert orch._should_skip(StageName.RESPONSE_GENERATION, ctx) is True

    def test_security_blocked_skips_remaining(self) -> None:
        ctx = PipelineContext(request_id="term-2", user_query="test")
        ctx.security_result = SecurityResult(
            status=SecurityStatus.BLOCK,
            threats=(ThreatType.INJECTION,),
            reason="bad",
        )

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        assert orch._should_skip(StageName.TASK_ANALYZER, ctx) is True
        assert orch._should_skip(StageName.SEMANTIC_CACHE, ctx) is True

    def test_cache_hit_skips_post_cache_stages(self) -> None:
        ctx = PipelineContext(request_id="term-3", user_query="test")
        ctx.cache_hit = True
        ctx.cache_response_text = "cached answer"

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        assert orch._should_skip(StageName.CONTEXT_MANAGER, ctx) is True
        assert orch._should_skip(StageName.RESPONSE_GENERATION, ctx) is True
        assert orch._should_skip(StageName.CACHE_STORE, ctx) is True
        # Cache lookup itself should NOT be skipped
        assert orch._should_skip(StageName.TASK_ANALYZER, ctx) is False

    def test_check_terminal_fast_response(self) -> None:
        ctx = PipelineContext(request_id="ct-1", user_query="hi")
        ctx.fast_detector_result = FastDetectorResult(
            is_fast_path=True, reason="greeting", canned_response="Hello!"
        )

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        result = orch._check_terminal(StageName.FAST_DETECTOR, ctx)
        assert result == PipelineEvent.FAST_RESPONSE

    def test_check_terminal_security_blocked(self) -> None:
        ctx = PipelineContext(request_id="ct-2", user_query="bad")
        ctx.security_result = SecurityResult(
            status=SecurityStatus.BLOCK,
            threats=(ThreatType.INJECTION,),
            reason="blocked",
        )

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        result = orch._check_terminal(StageName.SECURITY, ctx)
        assert result == PipelineEvent.SECURITY_BLOCKED

    def test_check_terminal_cache_hit(self) -> None:
        ctx = PipelineContext(request_id="ct-3", user_query="q")
        ctx.cache_hit = True

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        result = orch._check_terminal(StageName.SEMANTIC_CACHE, ctx)
        assert result == PipelineEvent.CACHE_HIT

    def test_check_terminal_returns_none_for_normal_stage(self) -> None:
        ctx = PipelineContext(request_id="ct-4", user_query="q")
        ctx.fast_detector_result = FastDetectorResult(
            is_fast_path=False, reason="normal"
        )

        from backend.app.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.STAGE_ORDER = PipelineOrchestrator.STAGE_ORDER

        result = orch._check_terminal(StageName.FAST_DETECTOR, ctx)
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineExceptions:
    def test_stage_execution_error_carries_stage(self) -> None:
        err = StageExecutionError("failed", stage="security")
        assert err.stage == "security"
        assert "failed" in str(err)

    def test_pipeline_error_base(self) -> None:
        err = PipelineError("base error")
        assert err.stage is None
        assert "base error" in str(err)

    def test_checkpoint_error(self) -> None:
        err = CheckpointError("save failed", stage="budget")
        assert err.stage == "budget"

    def test_stage_timeout_error(self) -> None:
        err = StageTimeoutError("timed out", stage="response_generation")
        assert err.stage == "response_generation"

    def test_pipeline_terminated(self) -> None:
        err = PipelineTerminated("early exit", stage="fast_detector")
        assert err.stage == "fast_detector"

    def test_all_exceptions_inherit_pipeline_error(self) -> None:
        for exc_cls in (StageExecutionError, CheckpointError, StageTimeoutError, PipelineTerminated):
            err = exc_cls("test")
            assert isinstance(err, PipelineError)
            assert isinstance(err, Exception)
