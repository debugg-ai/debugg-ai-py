import random
from collections import Counter
from unittest import mock

import pytest

import debugg_ai_sdk
from debugg_ai_sdk import start_span, start_transaction, capture_exception
from debugg_ai_sdk.tracing import Transaction
from debugg_ai_sdk.tracing_utils import Baggage
from debugg_ai_sdk.utils import logger


def test_sampling_decided_only_for_transactions(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=0.5)

    with start_transaction(name="hi") as transaction:
        assert transaction.sampled is not None

        with start_span() as span:
            assert span.sampled == transaction.sampled

    with start_span() as span:
        assert span.sampled is None


@pytest.mark.parametrize("sampled", [True, False])
def test_nested_transaction_sampling_override(debugg_ai_init, sampled):
    debugg_ai_init(traces_sample_rate=1.0)

    with start_transaction(name="outer", sampled=sampled) as outer_transaction:
        assert outer_transaction.sampled is sampled
        with start_transaction(
            name="inner", sampled=(not sampled)
        ) as inner_transaction:
            assert inner_transaction.sampled is not sampled
        assert outer_transaction.sampled is sampled


def test_no_double_sampling(debugg_ai_init, capture_events):
    # Transactions should not be subject to the global/error sample rate.
    # Only the traces_sample_rate should apply.
    debugg_ai_init(traces_sample_rate=1.0, sample_rate=0.0)
    events = capture_events()

    with start_transaction(name="/"):
        pass

    assert len(events) == 1


@pytest.mark.parametrize("sampling_decision", [True, False])
def test_get_transaction_and_span_from_scope_regardless_of_sampling_decision(
    debugg_ai_init, sampling_decision
):
    debugg_ai_init(traces_sample_rate=1.0)

    with start_transaction(name="/", sampled=sampling_decision):
        with start_span(op="child-span"):
            with start_span(op="child-child-span"):
                scope = debugg_ai_sdk.get_current_scope()
                assert scope.span.op == "child-child-span"
                assert scope.transaction.name == "/"


@pytest.mark.parametrize(
    "traces_sample_rate,expected_decision",
    [(0.0, False), (0.25, False), (0.75, True), (1.00, True)],
)
def test_uses_traces_sample_rate_correctly(
    debugg_ai_init,
    traces_sample_rate,
    expected_decision,
):
    debugg_ai_init(traces_sample_rate=traces_sample_rate)

    baggage = Baggage(debugg_ai_items={"sample_rand": "0.500000"})
    transaction = start_transaction(name="dogpark", baggage=baggage)
    assert transaction.sampled is expected_decision


@pytest.mark.parametrize(
    "traces_sampler_return_value,expected_decision",
    [(0.0, False), (0.25, False), (0.75, True), (1.00, True)],
)
def test_uses_traces_sampler_return_value_correctly(
    debugg_ai_init,
    traces_sampler_return_value,
    expected_decision,
):
    debugg_ai_init(traces_sampler=mock.Mock(return_value=traces_sampler_return_value))

    baggage = Baggage(debugg_ai_items={"sample_rand": "0.500000"})
    transaction = start_transaction(name="dogpark", baggage=baggage)
    assert transaction.sampled is expected_decision


@pytest.mark.parametrize("traces_sampler_return_value", [True, False])
def test_tolerates_traces_sampler_returning_a_boolean(
    debugg_ai_init, traces_sampler_return_value
):
    debugg_ai_init(traces_sampler=mock.Mock(return_value=traces_sampler_return_value))

    transaction = start_transaction(name="dogpark")
    assert transaction.sampled is traces_sampler_return_value


@pytest.mark.parametrize("sampling_decision", [True, False])
def test_only_captures_transaction_when_sampled_is_true(
    debugg_ai_init, sampling_decision, capture_events
):
    debugg_ai_init(traces_sampler=mock.Mock(return_value=sampling_decision))
    events = capture_events()

    transaction = start_transaction(name="dogpark")
    transaction.finish()

    assert len(events) == (1 if sampling_decision else 0)


@pytest.mark.parametrize(
    "traces_sample_rate,traces_sampler_return_value", [(0, True), (1, False)]
)
def test_prefers_traces_sampler_to_traces_sample_rate(
    debugg_ai_init,
    traces_sample_rate,
    traces_sampler_return_value,
):
    # make traces_sample_rate imply the opposite of traces_sampler, to prove
    # that traces_sampler takes precedence
    traces_sampler = mock.Mock(return_value=traces_sampler_return_value)
    debugg_ai_init(
        traces_sample_rate=traces_sample_rate,
        traces_sampler=traces_sampler,
    )

    transaction = start_transaction(name="dogpark")
    assert traces_sampler.called is True
    assert transaction.sampled is traces_sampler_return_value


@pytest.mark.parametrize("parent_sampling_decision", [True, False])
def test_ignores_inherited_sample_decision_when_traces_sampler_defined(
    debugg_ai_init, parent_sampling_decision
):
    # make traces_sampler pick the opposite of the inherited decision, to prove
    # that traces_sampler takes precedence
    traces_sampler = mock.Mock(return_value=not parent_sampling_decision)
    debugg_ai_init(traces_sampler=traces_sampler)

    transaction = start_transaction(
        name="dogpark", parent_sampled=parent_sampling_decision
    )
    assert transaction.sampled is not parent_sampling_decision


@pytest.mark.parametrize("explicit_decision", [True, False])
def test_traces_sampler_doesnt_overwrite_explicitly_passed_sampling_decision(
    debugg_ai_init, explicit_decision
):
    # make traces_sampler pick the opposite of the explicit decision, to prove
    # that the explicit decision takes precedence
    traces_sampler = mock.Mock(return_value=not explicit_decision)
    debugg_ai_init(traces_sampler=traces_sampler)

    transaction = start_transaction(name="dogpark", sampled=explicit_decision)
    assert transaction.sampled is explicit_decision


@pytest.mark.parametrize("parent_sampling_decision", [True, False])
def test_inherits_parent_sampling_decision_when_traces_sampler_undefined(
    debugg_ai_init, parent_sampling_decision
):
    # make sure the parent sampling decision is the opposite of what
    # traces_sample_rate would produce, to prove the inheritance takes
    # precedence
    debugg_ai_init(traces_sample_rate=0.5)
    mock_random_value = 0.25 if parent_sampling_decision is False else 0.75

    with mock.patch.object(random, "random", return_value=mock_random_value):
        transaction = start_transaction(
            name="dogpark", parent_sampled=parent_sampling_decision
        )
        assert transaction.sampled is parent_sampling_decision


@pytest.mark.parametrize("parent_sampling_decision", [True, False])
def test_passes_parent_sampling_decision_in_sampling_context(
    debugg_ai_init, parent_sampling_decision
):
    debugg_ai_init(traces_sample_rate=1.0)

    debugg_ai_trace_header = (
        "12312012123120121231201212312012-1121201211212012-{sampled}".format(
            sampled=int(parent_sampling_decision)
        )
    )

    transaction = Transaction.continue_from_headers(
        headers={"debugg-ai-trace": debugg_ai_trace_header}, name="dogpark"
    )

    def mock_set_initial_sampling_decision(_, sampling_context):
        assert "parent_sampled" in sampling_context
        assert sampling_context["parent_sampled"] is parent_sampling_decision

    with mock.patch(
        "debugg_ai_sdk.tracing.Transaction._set_initial_sampling_decision",
        mock_set_initial_sampling_decision,
    ):
        start_transaction(transaction=transaction)


def test_passes_custom_sampling_context_from_start_transaction_to_traces_sampler(
    debugg_ai_init, DictionaryContaining  # noqa: N803
):
    traces_sampler = mock.Mock()
    debugg_ai_init(traces_sampler=traces_sampler)

    start_transaction(custom_sampling_context={"dogs": "yes", "cats": "maybe"})

    traces_sampler.assert_any_call(
        DictionaryContaining({"dogs": "yes", "cats": "maybe"})
    )


def test_sample_rate_affects_errors(debugg_ai_init, capture_events):
    debugg_ai_init(sample_rate=0)
    events = capture_events()

    try:
        1 / 0
    except Exception:
        capture_exception()

    assert len(events) == 0


@pytest.mark.parametrize(
    "traces_sampler_return_value",
    [
        "dogs are great",  # wrong type
        (0, 1),  # wrong type
        {"Maisey": "Charllie"},  # wrong type
        [True, True],  # wrong type
        {0.2012},  # wrong type
        float("NaN"),  # wrong type
        None,  # wrong type
        -1.121,  # wrong value
        1.231,  # wrong value
    ],
)
def test_warns_and_sets_sampled_to_false_on_invalid_traces_sampler_return_value(
    debugg_ai_init, traces_sampler_return_value, StringContaining  # noqa: N803
):
    debugg_ai_init(traces_sampler=mock.Mock(return_value=traces_sampler_return_value))

    with mock.patch.object(logger, "warning", mock.Mock()):
        transaction = start_transaction(name="dogpark")
        logger.warning.assert_any_call(StringContaining("Given sample rate is invalid"))
        assert transaction.sampled is False


@pytest.mark.parametrize(
    "traces_sample_rate,sampled_output,expected_record_lost_event_calls",
    [
        (None, False, []),
        (
            0.0,
            False,
            [("sample_rate", "transaction", None, 1), ("sample_rate", "span", None, 1)],
        ),
        (1.0, True, []),
    ],
)
def test_records_lost_event_only_if_traces_sample_rate_enabled(
    debugg_ai_init,
    capture_record_lost_event_calls,
    traces_sample_rate,
    sampled_output,
    expected_record_lost_event_calls,
):
    debugg_ai_init(traces_sample_rate=traces_sample_rate)
    record_lost_event_calls = capture_record_lost_event_calls()

    transaction = start_transaction(name="dogpark")
    assert transaction.sampled is sampled_output
    transaction.finish()

    # Use Counter because order of calls does not matter
    assert Counter(record_lost_event_calls) == Counter(expected_record_lost_event_calls)


@pytest.mark.parametrize(
    "traces_sampler,sampled_output,expected_record_lost_event_calls",
    [
        (None, False, []),
        (
            lambda _x: 0.0,
            False,
            [("sample_rate", "transaction", None, 1), ("sample_rate", "span", None, 1)],
        ),
        (lambda _x: 1.0, True, []),
    ],
)
def test_records_lost_event_only_if_traces_sampler_enabled(
    debugg_ai_init,
    capture_record_lost_event_calls,
    traces_sampler,
    sampled_output,
    expected_record_lost_event_calls,
):
    debugg_ai_init(traces_sampler=traces_sampler)
    record_lost_event_calls = capture_record_lost_event_calls()

    transaction = start_transaction(name="dogpark")
    assert transaction.sampled is sampled_output
    transaction.finish()

    # Use Counter because order of calls does not matter
    assert Counter(record_lost_event_calls) == Counter(expected_record_lost_event_calls)
