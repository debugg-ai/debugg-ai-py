import pytest

import debugg_ai_sdk


@pytest.fixture
def capture_spotlight_envelopes(monkeypatch):
    def inner():
        envelopes = []
        test_spotlight = debugg_ai_sdk.get_client().spotlight
        old_capture_envelope = test_spotlight.capture_envelope

        def append_envelope(envelope):
            envelopes.append(envelope)
            return old_capture_envelope(envelope)

        monkeypatch.setattr(test_spotlight, "capture_envelope", append_envelope)
        return envelopes

    return inner


def test_spotlight_off_by_default(debugg_ai_init):
    debugg_ai_init()
    assert debugg_ai_sdk.get_client().spotlight is None


def test_spotlight_default_url(debugg_ai_init):
    debugg_ai_init(spotlight=True)

    spotlight = debugg_ai_sdk.get_client().spotlight
    assert spotlight is not None
    assert spotlight.url == "http://localhost:8969/stream"


def test_spotlight_custom_url(debugg_ai_init):
    debugg_ai_init(spotlight="http://foobar@test.com/132")

    spotlight = debugg_ai_sdk.get_client().spotlight
    assert spotlight is not None
    assert spotlight.url == "http://foobar@test.com/132"


def test_spotlight_envelope(debugg_ai_init, capture_spotlight_envelopes):
    debugg_ai_init(spotlight=True)
    envelopes = capture_spotlight_envelopes()

    try:
        raise ValueError("aha!")
    except Exception:
        debugg_ai_sdk.capture_exception()

    (envelope,) = envelopes
    payload = envelope.items[0].payload.json

    assert payload["exception"]["values"][0]["value"] == "aha!"
