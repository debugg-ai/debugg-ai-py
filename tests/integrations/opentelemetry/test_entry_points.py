import importlib
import os
from unittest.mock import patch

from opentelemetry import propagate
from debugg_ai_sdk.integrations.opentelemetry import DebuggAIPropagator


def test_propagator_loaded_if_mentioned_in_environment_variable():
    try:
        with patch.dict(os.environ, {"OTEL_PROPAGATORS": "debugg-ai"}):
            importlib.reload(propagate)

            assert len(propagate.propagators) == 1
            assert isinstance(propagate.propagators[0], DebuggAIPropagator)
    finally:
        importlib.reload(propagate)
