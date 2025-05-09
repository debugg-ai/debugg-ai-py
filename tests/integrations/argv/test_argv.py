import sys

from debugg_ai_sdk import capture_message
from debugg_ai_sdk.integrations.argv import ArgvIntegration


def test_basic(debugg_ai_init, capture_events, monkeypatch):
    debugg_ai_init(integrations=[ArgvIntegration()])

    argv = ["foo", "bar", "baz"]
    monkeypatch.setattr(sys, "argv", argv)

    events = capture_events()
    capture_message("hi")
    (event,) = events
    assert event["extra"]["sys.argv"] == argv
