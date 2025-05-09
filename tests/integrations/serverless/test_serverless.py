import pytest

from debugg_ai_sdk.integrations.serverless import serverless_function


def test_basic(debugg_ai_init, capture_exceptions, monkeypatch):
    debugg_ai_init()
    exceptions = capture_exceptions()

    flush_calls = []

    @serverless_function
    def foo():
        monkeypatch.setattr("debugg_ai_sdk.flush", lambda: flush_calls.append(1))
        1 / 0

    with pytest.raises(ZeroDivisionError):
        foo()

    (exception,) = exceptions
    assert isinstance(exception, ZeroDivisionError)

    assert flush_calls == [1]


def test_flush_disabled(debugg_ai_init, capture_exceptions, monkeypatch):
    debugg_ai_init()
    exceptions = capture_exceptions()

    flush_calls = []

    monkeypatch.setattr("debugg_ai_sdk.flush", lambda: flush_calls.append(1))

    @serverless_function(flush=False)
    def foo():
        1 / 0

    with pytest.raises(ZeroDivisionError):
        foo()

    (exception,) = exceptions
    assert isinstance(exception, ZeroDivisionError)

    assert flush_calls == []
