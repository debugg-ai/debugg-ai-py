import debugg_ai_sdk
from debugg_ai_sdk.tracing import NoOpSpan

# These tests make sure that the examples from the documentation [1]
# are working when OTel (OpenTelemetry) instrumentation is turned on,
# and therefore, the DebuggAI tracing should not do anything.
#
# 1: https://docs.debugg.ai/platforms/python/performance/instrumentation/custom-instrumentation/


def test_noop_start_transaction(debugg_ai_init):
    debugg_ai_init(instrumenter="otel")

    with debugg_ai_sdk.start_transaction(
        op="task", name="test_transaction_name"
    ) as transaction:
        assert isinstance(transaction, NoOpSpan)
        assert debugg_ai_sdk.get_current_scope().span is transaction

        transaction.name = "new name"


def test_noop_start_span(debugg_ai_init):
    debugg_ai_init(instrumenter="otel")

    with debugg_ai_sdk.start_span(op="http", name="GET /") as span:
        assert isinstance(span, NoOpSpan)
        assert debugg_ai_sdk.get_current_scope().span is span

        span.set_tag("http.response.status_code", 418)
        span.set_data("http.entity_type", "teapot")


def test_noop_transaction_start_child(debugg_ai_init):
    debugg_ai_init(instrumenter="otel")

    transaction = debugg_ai_sdk.start_transaction(name="task")
    assert isinstance(transaction, NoOpSpan)

    with transaction.start_child(op="child_task") as child:
        assert isinstance(child, NoOpSpan)
        assert debugg_ai_sdk.get_current_scope().span is child


def test_noop_span_start_child(debugg_ai_init):
    debugg_ai_init(instrumenter="otel")
    span = debugg_ai_sdk.start_span(name="task")
    assert isinstance(span, NoOpSpan)

    with span.start_child(op="child_task") as child:
        assert isinstance(child, NoOpSpan)
        assert debugg_ai_sdk.get_current_scope().span is child
