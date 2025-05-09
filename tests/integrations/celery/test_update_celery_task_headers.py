from copy import copy
import itertools
import pytest

from unittest import mock

from debugg_ai_sdk.integrations.celery import _update_celery_task_headers
import debugg_ai_sdk
from debugg_ai_sdk.tracing_utils import Baggage


BAGGAGE_VALUE = (
    "debugg-ai-trace_id=771a43a4192642f0b136d5159a501700,"
    "debugg-ai-public_key=49d0f7386ad645858ae85020e393bef3,"
    "debugg-ai-sample_rate=0.1337,"
    "custom=value"
)

DEBUGG_AI_TRACE_VALUE = "771a43a4192642f0b136d5159a501700-1234567890abcdef-1"


@pytest.mark.parametrize("monitor_beat_tasks", [True, False, None, "", "bla", 1, 0])
def test_monitor_beat_tasks(monitor_beat_tasks):
    headers = {}
    span = None

    outgoing_headers = _update_celery_task_headers(headers, span, monitor_beat_tasks)

    assert headers == {}  # left unchanged

    if monitor_beat_tasks:
        assert outgoing_headers["debugg-ai-monitor-start-timestamp-s"] == mock.ANY
        assert (
            outgoing_headers["headers"]["debugg-ai-monitor-start-timestamp-s"] == mock.ANY
        )
    else:
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers["headers"]


@pytest.mark.parametrize("monitor_beat_tasks", [True, False, None, "", "bla", 1, 0])
def test_monitor_beat_tasks_with_headers(monitor_beat_tasks):
    headers = {
        "blub": "foo",
        "debugg-ai-something": "bar",
        "debugg-ai-task-enqueued-time": mock.ANY,
    }
    span = None

    outgoing_headers = _update_celery_task_headers(headers, span, monitor_beat_tasks)

    assert headers == {
        "blub": "foo",
        "debugg-ai-something": "bar",
        "debugg-ai-task-enqueued-time": mock.ANY,
    }  # left unchanged

    if monitor_beat_tasks:
        assert outgoing_headers["blub"] == "foo"
        assert outgoing_headers["debugg-ai-something"] == "bar"
        assert outgoing_headers["debugg-ai-monitor-start-timestamp-s"] == mock.ANY
        assert outgoing_headers["headers"]["debugg-ai-something"] == "bar"
        assert (
            outgoing_headers["headers"]["debugg-ai-monitor-start-timestamp-s"] == mock.ANY
        )
    else:
        assert outgoing_headers["blub"] == "foo"
        assert outgoing_headers["debugg-ai-something"] == "bar"
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers["headers"]


def test_span_with_transaction(debugg_ai_init):
    debugg_ai_init(enable_tracing=True)
    headers = {}
    monitor_beat_tasks = False

    with debugg_ai_sdk.start_transaction(name="test_transaction") as transaction:
        with debugg_ai_sdk.start_span(op="test_span") as span:
            outgoing_headers = _update_celery_task_headers(
                headers, span, monitor_beat_tasks
            )

            assert outgoing_headers["debugg-ai-trace"] == span.to_traceparent()
            assert outgoing_headers["headers"]["debugg-ai-trace"] == span.to_traceparent()
            assert outgoing_headers["baggage"] == transaction.get_baggage().serialize()
            assert (
                outgoing_headers["headers"]["baggage"]
                == transaction.get_baggage().serialize()
            )


def test_span_with_transaction_custom_headers(debugg_ai_init):
    debugg_ai_init(enable_tracing=True)
    headers = {
        "baggage": BAGGAGE_VALUE,
        "debugg-ai-trace": DEBUGG_AI_TRACE_VALUE,
    }

    with debugg_ai_sdk.start_transaction(name="test_transaction") as transaction:
        with debugg_ai_sdk.start_span(op="test_span") as span:
            outgoing_headers = _update_celery_task_headers(headers, span, False)

            assert outgoing_headers["debugg-ai-trace"] == span.to_traceparent()
            assert outgoing_headers["headers"]["debugg-ai-trace"] == span.to_traceparent()

            incoming_baggage = Baggage.from_incoming_header(headers["baggage"])
            combined_baggage = copy(transaction.get_baggage())
            combined_baggage.debugg_ai_items.update(incoming_baggage.debugg_ai_items)
            combined_baggage.third_party_items = ",".join(
                [
                    x
                    for x in [
                        combined_baggage.third_party_items,
                        incoming_baggage.third_party_items,
                    ]
                    if x is not None and x != ""
                ]
            )
            assert outgoing_headers["baggage"] == combined_baggage.serialize(
                include_third_party=True
            )
            assert outgoing_headers["headers"]["baggage"] == combined_baggage.serialize(
                include_third_party=True
            )


@pytest.mark.parametrize("monitor_beat_tasks", [True, False])
def test_celery_trace_propagation_default(debugg_ai_init, monitor_beat_tasks):
    """
    The celery integration does not check the traces_sample_rate.
    By default traces_sample_rate is None which means "do not propagate traces".
    But the celery integration does not check this value.
    The Celery integration has its own mechanism to propagate traces:
    https://docs.debugg.ai/platforms/python/integrations/celery/#distributed-traces
    """
    debugg_ai_init()

    headers = {}
    span = None

    scope = debugg_ai_sdk.get_isolation_scope()

    outgoing_headers = _update_celery_task_headers(headers, span, monitor_beat_tasks)

    assert outgoing_headers["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["headers"]["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["baggage"] == scope.get_baggage().serialize()
    assert outgoing_headers["headers"]["baggage"] == scope.get_baggage().serialize()

    if monitor_beat_tasks:
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers["headers"]
    else:
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers["headers"]


@pytest.mark.parametrize(
    "traces_sample_rate,monitor_beat_tasks",
    list(itertools.product([None, 0, 0.0, 0.5, 1.0, 1, 2], [True, False])),
)
def test_celery_trace_propagation_traces_sample_rate(
    debugg_ai_init, traces_sample_rate, monitor_beat_tasks
):
    """
    The celery integration does not check the traces_sample_rate.
    By default traces_sample_rate is None which means "do not propagate traces".
    But the celery integration does not check this value.
    The Celery integration has its own mechanism to propagate traces:
    https://docs.debugg.ai/platforms/python/integrations/celery/#distributed-traces
    """
    debugg_ai_init(traces_sample_rate=traces_sample_rate)

    headers = {}
    span = None

    scope = debugg_ai_sdk.get_isolation_scope()

    outgoing_headers = _update_celery_task_headers(headers, span, monitor_beat_tasks)

    assert outgoing_headers["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["headers"]["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["baggage"] == scope.get_baggage().serialize()
    assert outgoing_headers["headers"]["baggage"] == scope.get_baggage().serialize()

    if monitor_beat_tasks:
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers["headers"]
    else:
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers["headers"]


@pytest.mark.parametrize(
    "enable_tracing,monitor_beat_tasks",
    list(itertools.product([None, True, False], [True, False])),
)
def test_celery_trace_propagation_enable_tracing(
    debugg_ai_init, enable_tracing, monitor_beat_tasks
):
    """
    The celery integration does not check the traces_sample_rate.
    By default traces_sample_rate is None which means "do not propagate traces".
    But the celery integration does not check this value.
    The Celery integration has its own mechanism to propagate traces:
    https://docs.debugg.ai/platforms/python/integrations/celery/#distributed-traces
    """
    debugg_ai_init(enable_tracing=enable_tracing)

    headers = {}
    span = None

    scope = debugg_ai_sdk.get_isolation_scope()

    outgoing_headers = _update_celery_task_headers(headers, span, monitor_beat_tasks)

    assert outgoing_headers["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["headers"]["debugg-ai-trace"] == scope.get_traceparent()
    assert outgoing_headers["baggage"] == scope.get_baggage().serialize()
    assert outgoing_headers["headers"]["baggage"] == scope.get_baggage().serialize()

    if monitor_beat_tasks:
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" in outgoing_headers["headers"]
    else:
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers
        assert "debugg-ai-monitor-start-timestamp-s" not in outgoing_headers["headers"]
