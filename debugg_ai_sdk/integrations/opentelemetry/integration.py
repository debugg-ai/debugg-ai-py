"""
IMPORTANT: The contents of this file are part of a proof of concept and as such
are experimental and not suitable for production use. They may be changed or
removed at any time without prior notice.
"""

from debugg_ai_sdk.integrations import DidNotEnable, Integration
from debugg_ai_sdk.integrations.opentelemetry.propagator import DebuggAIPropagator
from debugg_ai_sdk.integrations.opentelemetry.span_processor import DebuggAISpanProcessor
from debugg_ai_sdk.utils import logger

try:
    from opentelemetry import trace
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.sdk.trace import TracerProvider
except ImportError:
    raise DidNotEnable("opentelemetry not installed")

try:
    from opentelemetry.instrumentation.django import DjangoInstrumentor  # type: ignore[import-not-found]
except ImportError:
    DjangoInstrumentor = None


CONFIGURABLE_INSTRUMENTATIONS = {
    DjangoInstrumentor: {"is_sql_commentor_enabled": True},
}


class OpenTelemetryIntegration(Integration):
    identifier = "opentelemetry"

    @staticmethod
    def setup_once():
        # type: () -> None
        logger.warning(
            "[OTel] Initializing highly experimental OpenTelemetry support. "
            "Use at your own risk."
        )

        _setup_debugg_ai_tracing()
        # _setup_instrumentors()

        logger.debug("[OTel] Finished setting up OpenTelemetry integration")


def _setup_debugg_ai_tracing():
    # type: () -> None
    provider = TracerProvider()
    provider.add_span_processor(DebuggAISpanProcessor())
    trace.set_tracer_provider(provider)
    set_global_textmap(DebuggAIPropagator())


def _setup_instrumentors():
    # type: () -> None
    for instrumentor, kwargs in CONFIGURABLE_INSTRUMENTATIONS.items():
        instrumentor().instrument(**kwargs)
