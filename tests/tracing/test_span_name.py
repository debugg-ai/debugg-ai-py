import pytest

import debugg_ai_sdk


def test_start_span_description(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="hi"):
        with pytest.deprecated_call():
            with debugg_ai_sdk.start_span(op="foo", description="span-desc"):
                ...

    (event,) = events

    assert event["spans"][0]["description"] == "span-desc"


def test_start_span_name(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="hi"):
        with debugg_ai_sdk.start_span(op="foo", name="span-name"):
            ...

    (event,) = events

    assert event["spans"][0]["description"] == "span-name"


def test_start_child_description(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="hi"):
        with pytest.deprecated_call():
            with debugg_ai_sdk.start_span(op="foo", description="span-desc") as span:
                with span.start_child(op="bar", description="child-desc"):
                    ...

    (event,) = events

    assert event["spans"][-1]["description"] == "child-desc"


def test_start_child_name(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="hi"):
        with debugg_ai_sdk.start_span(op="foo", name="span-name") as span:
            with span.start_child(op="bar", name="child-name"):
                ...

    (event,) = events

    assert event["spans"][-1]["description"] == "child-name"
