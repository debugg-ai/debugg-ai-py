import pytest

from unittest import mock
from unittest.mock import MagicMock

from opentelemetry.context import get_current
from opentelemetry.trace import (
    SpanContext,
    TraceFlags,
    set_span_in_context,
)
from opentelemetry.trace.propagation import get_current_span

from debugg_ai_sdk.integrations.opentelemetry.consts import (
    DEBUGG_AI_BAGGAGE_KEY,
    DEBUGG_AI_TRACE_KEY,
)
from debugg_ai_sdk.integrations.opentelemetry.propagator import DebuggAIPropagator
from debugg_ai_sdk.integrations.opentelemetry.span_processor import DebuggAISpanProcessor
from debugg_ai_sdk.tracing_utils import Baggage


@pytest.mark.forked
def test_extract_no_context_no_debugg_ai_trace_header():
    """
    No context and NO DebuggAI trace data in getter.
    Extract should return empty context.
    """
    carrier = None
    context = None
    getter = MagicMock()
    getter.get.return_value = None

    modified_context = DebuggAIPropagator().extract(carrier, context, getter)

    assert modified_context == {}


@pytest.mark.forked
def test_extract_context_no_debugg_ai_trace_header():
    """
    Context but NO DebuggAI trace data in getter.
    Extract should return context as is.
    """
    carrier = None
    context = {"some": "value"}
    getter = MagicMock()
    getter.get.return_value = None

    modified_context = DebuggAIPropagator().extract(carrier, context, getter)

    assert modified_context == context


@pytest.mark.forked
def test_extract_empty_context_debugg_ai_trace_header_no_baggage():
    """
    Empty context but DebuggAI trace data but NO Baggage in getter.
    Extract should return context that has empty baggage in it and also a NoopSpan with span_id and trace_id.
    """
    carrier = None
    context = {}
    getter = MagicMock()
    getter.get.side_effect = [
        ["1234567890abcdef1234567890abcdef-1234567890abcdef-1"],
        None,
    ]

    modified_context = DebuggAIPropagator().extract(carrier, context, getter)

    assert len(modified_context.keys()) == 3

    assert modified_context[DEBUGG_AI_TRACE_KEY] == {
        "trace_id": "1234567890abcdef1234567890abcdef",
        "parent_span_id": "1234567890abcdef",
        "parent_sampled": True,
    }
    assert modified_context[DEBUGG_AI_BAGGAGE_KEY].serialize() == ""

    span_context = get_current_span(modified_context).get_span_context()
    assert span_context.span_id == int("1234567890abcdef", 16)
    assert span_context.trace_id == int("1234567890abcdef1234567890abcdef", 16)


@pytest.mark.forked
def test_extract_context_debugg_ai_trace_header_baggage():
    """
    Empty context but DebuggAI trace data and Baggage in getter.
    Extract should return context that has baggage in it and also a NoopSpan with span_id and trace_id.
    """
    baggage_header = (
        "other-vendor-value-1=foo;bar;baz, debugg-ai-trace_id=771a43a4192642f0b136d5159a501700, "
        "debugg-ai-public_key=49d0f7386ad645858ae85020e393bef3, debugg-ai-sample_rate=0.01337, "
        "debugg-ai-user_id=Am%C3%A9lie, other-vendor-value-2=foo;bar;"
    )

    carrier = None
    context = {"some": "value"}
    getter = MagicMock()
    getter.get.side_effect = [
        ["1234567890abcdef1234567890abcdef-1234567890abcdef-1"],
        [baggage_header],
    ]

    modified_context = DebuggAIPropagator().extract(carrier, context, getter)

    assert len(modified_context.keys()) == 4

    assert modified_context[DEBUGG_AI_TRACE_KEY] == {
        "trace_id": "1234567890abcdef1234567890abcdef",
        "parent_span_id": "1234567890abcdef",
        "parent_sampled": True,
    }

    assert modified_context[DEBUGG_AI_BAGGAGE_KEY].serialize() == (
        "debugg-ai-trace_id=771a43a4192642f0b136d5159a501700,"
        "debugg-ai-public_key=49d0f7386ad645858ae85020e393bef3,"
        "debugg-ai-sample_rate=0.01337,debugg-ai-user_id=Am%C3%A9lie"
    )

    span_context = get_current_span(modified_context).get_span_context()
    assert span_context.span_id == int("1234567890abcdef", 16)
    assert span_context.trace_id == int("1234567890abcdef1234567890abcdef", 16)


@pytest.mark.forked
def test_inject_empty_otel_span_map():
    """
    Empty otel_span_map.
    So there is no debugg_ai_span to be found in inject()
    and the function is returned early and no setters are called.
    """
    carrier = None
    context = get_current()
    setter = MagicMock()
    setter.set = MagicMock()

    span_context = SpanContext(
        trace_id=int("1234567890abcdef1234567890abcdef", 16),
        span_id=int("1234567890abcdef", 16),
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True,
    )
    span = MagicMock()
    span.get_span_context.return_value = span_context

    with mock.patch(
        "debugg_ai_sdk.integrations.opentelemetry.propagator.trace.get_current_span",
        return_value=span,
    ):
        full_context = set_span_in_context(span, context)
        DebuggAIPropagator().inject(carrier, full_context, setter)

        setter.set.assert_not_called()


@pytest.mark.forked
def test_inject_debugg_ai_span_no_baggage():
    """
    Inject a debugg-ai span with no baggage.
    """
    carrier = None
    context = get_current()
    setter = MagicMock()
    setter.set = MagicMock()

    trace_id = "1234567890abcdef1234567890abcdef"
    span_id = "1234567890abcdef"

    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True,
    )
    span = MagicMock()
    span.get_span_context.return_value = span_context

    debugg_ai_span = MagicMock()
    debugg_ai_span.to_traceparent = mock.Mock(
        return_value="1234567890abcdef1234567890abcdef-1234567890abcdef-1"
    )
    debugg_ai_span.containing_transaction.get_baggage = mock.Mock(return_value=None)

    span_processor = DebuggAISpanProcessor()
    span_processor.otel_span_map[span_id] = debugg_ai_span

    with mock.patch(
        "debugg_ai_sdk.integrations.opentelemetry.propagator.trace.get_current_span",
        return_value=span,
    ):
        full_context = set_span_in_context(span, context)
        DebuggAIPropagator().inject(carrier, full_context, setter)

        setter.set.assert_called_once_with(
            carrier,
            "debugg-ai-trace",
            "1234567890abcdef1234567890abcdef-1234567890abcdef-1",
        )


def test_inject_debugg_ai_span_empty_baggage():
    """
    Inject a debugg-ai span with no baggage.
    """
    carrier = None
    context = get_current()
    setter = MagicMock()
    setter.set = MagicMock()

    trace_id = "1234567890abcdef1234567890abcdef"
    span_id = "1234567890abcdef"

    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True,
    )
    span = MagicMock()
    span.get_span_context.return_value = span_context

    debugg_ai_span = MagicMock()
    debugg_ai_span.to_traceparent = mock.Mock(
        return_value="1234567890abcdef1234567890abcdef-1234567890abcdef-1"
    )
    debugg_ai_span.containing_transaction.get_baggage = mock.Mock(return_value=Baggage({}))

    span_processor = DebuggAISpanProcessor()
    span_processor.otel_span_map[span_id] = debugg_ai_span

    with mock.patch(
        "debugg_ai_sdk.integrations.opentelemetry.propagator.trace.get_current_span",
        return_value=span,
    ):
        full_context = set_span_in_context(span, context)
        DebuggAIPropagator().inject(carrier, full_context, setter)

        setter.set.assert_called_once_with(
            carrier,
            "debugg-ai-trace",
            "1234567890abcdef1234567890abcdef-1234567890abcdef-1",
        )


def test_inject_debugg_ai_span_baggage():
    """
    Inject a debugg-ai span with baggage.
    """
    carrier = None
    context = get_current()
    setter = MagicMock()
    setter.set = MagicMock()

    trace_id = "1234567890abcdef1234567890abcdef"
    span_id = "1234567890abcdef"

    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        is_remote=True,
    )
    span = MagicMock()
    span.get_span_context.return_value = span_context

    debugg_ai_span = MagicMock()
    debugg_ai_span.to_traceparent = mock.Mock(
        return_value="1234567890abcdef1234567890abcdef-1234567890abcdef-1"
    )
    debugg_ai_items = {
        "debugg-ai-trace_id": "771a43a4192642f0b136d5159a501700",
        "debugg-ai-public_key": "49d0f7386ad645858ae85020e393bef3",
        "debugg-ai-sample_rate": 0.01337,
        "debugg-ai-user_id": "Amélie",
    }
    baggage = Baggage(debugg_ai_items=debugg_ai_items)
    debugg_ai_span.containing_transaction.get_baggage = MagicMock(return_value=baggage)

    span_processor = DebuggAISpanProcessor()
    span_processor.otel_span_map[span_id] = debugg_ai_span

    with mock.patch(
        "debugg_ai_sdk.integrations.opentelemetry.propagator.trace.get_current_span",
        return_value=span,
    ):
        full_context = set_span_in_context(span, context)
        DebuggAIPropagator().inject(carrier, full_context, setter)

        setter.set.assert_any_call(
            carrier,
            "debugg-ai-trace",
            "1234567890abcdef1234567890abcdef-1234567890abcdef-1",
        )

        setter.set.assert_any_call(
            carrier,
            "baggage",
            baggage.serialize(),
        )
