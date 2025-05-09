import debugg_ai_sdk

from debugg_ai_sdk.integrations.modules import ModulesIntegration


def test_basic(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[ModulesIntegration()])
    events = capture_events()

    debugg_ai_sdk.capture_exception(ValueError())

    (event,) = events
    assert "debugg-ai-sdk" in event["modules"]
    assert "pytest" in event["modules"]
