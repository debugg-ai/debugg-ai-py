"""
These tests exist to verify that Scope.continue_trace() correctly propagates the
sample_rand value onto the transaction's baggage.

We check both the case where there is an incoming sample_rand, as well as the case
where we need to compute it because it is missing.
"""

from unittest import mock
from unittest.mock import Mock

import debugg_ai_sdk


def test_continue_trace_with_sample_rand():
    """
    Test that an incoming sample_rand is propagated onto the transaction's baggage.
    """
    headers = {
        "debugg-ai-trace": "00000000000000000000000000000000-0000000000000000-0",
        "baggage": "debugg-ai-sample_rand=0.1,debugg-ai-sample_rate=0.5",
    }

    transaction = debugg_ai_sdk.continue_trace(headers)
    assert transaction.get_baggage().debugg_ai_items["sample_rand"] == "0.1"


def test_continue_trace_missing_sample_rand():
    """
    Test that a missing sample_rand is filled in onto the transaction's baggage.
    """

    headers = {
        "debugg-ai-trace": "00000000000000000000000000000000-0000000000000000",
        "baggage": "debugg-ai-placeholder=asdf",
    }

    mock_uniform = Mock(return_value=0.5)

    with mock.patch("debugg_ai_sdk.tracing_utils.Random.uniform", mock_uniform):
        transaction = debugg_ai_sdk.continue_trace(headers)

    assert transaction.get_baggage().debugg_ai_items["sample_rand"] == "0.500000"
