import pytest
import debugg_ai_sdk


@pytest.fixture(autouse=True)
def isolate_hub(suppress_deprecation_warnings):
    with debugg_ai_sdk.Hub(None):
        yield
