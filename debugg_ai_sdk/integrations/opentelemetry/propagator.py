from opentelemetry import trace
from opentelemetry.context import (
    Context,
    get_current,
    set_value,
)
from opentelemetry.propagators.textmap import (
    CarrierT,
    Getter,
    Setter,
    TextMapPropagator,
    default_getter,
    default_setter,
)
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
)

from debugg_ai_sdk.integrations.opentelemetry.consts import (
    DEBUGG_AI_BAGGAGE_KEY,
    DEBUGG_AI_TRACE_KEY,
)
from debugg_ai_sdk.integrations.opentelemetry.span_processor import (
    DebuggAISpanProcessor,
)
from debugg_ai_sdk.tracing import (
    BAGGAGE_HEADER_NAME,
    DEBUGG_AI_TRACE_HEADER_NAME,
)
from debugg_ai_sdk.tracing_utils import Baggage, extract_debugg_ai_trace_data

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Set


class DebuggAIPropagator(TextMapPropagator):
    """
    Propagates tracing headers for DebuggAI's tracing system in a way OTel understands.
    """

    def extract(self, carrier, context=None, getter=default_getter):
        # type: (CarrierT, Optional[Context], Getter[CarrierT]) -> Context
        if context is None:
            context = get_current()

        debugg_ai_trace = getter.get(carrier, DEBUGG_AI_TRACE_HEADER_NAME)
        if not debugg_ai_trace:
            return context

        debugg_ai_trace = extract_debugg_ai_trace_data(debugg_ai_trace[0])
        if not debugg_ai_trace:
            return context

        context = set_value(DEBUGG_AI_TRACE_KEY, debugg_ai_trace, context)

        trace_id, span_id = debugg_ai_trace["trace_id"], debugg_ai_trace["parent_span_id"]

        span_context = SpanContext(
            trace_id=int(trace_id, 16),  # type: ignore
            span_id=int(span_id, 16),  # type: ignore
            # we simulate a sampled trace on the otel side and leave the sampling to debugg-ai
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            is_remote=True,
        )

        baggage_header = getter.get(carrier, BAGGAGE_HEADER_NAME)

        if baggage_header:
            baggage = Baggage.from_incoming_header(baggage_header[0])
        else:
            # If there's an incoming debugg-ai-trace but no incoming baggage header,
            # for instance in traces coming from older SDKs,
            # baggage will be empty and frozen and won't be populated as head SDK.
            baggage = Baggage(debugg_ai_items={})

        baggage.freeze()
        context = set_value(DEBUGG_AI_BAGGAGE_KEY, baggage, context)

        span = NonRecordingSpan(span_context)
        modified_context = trace.set_span_in_context(span, context)
        return modified_context

    def inject(self, carrier, context=None, setter=default_setter):
        # type: (CarrierT, Optional[Context], Setter[CarrierT]) -> None
        if context is None:
            context = get_current()

        current_span = trace.get_current_span(context)
        current_span_context = current_span.get_span_context()

        if not current_span_context.is_valid:
            return

        span_id = trace.format_span_id(current_span_context.span_id)

        span_map = DebuggAISpanProcessor().otel_span_map
        debugg_ai_span = span_map.get(span_id, None)
        if not debugg_ai_span:
            return

        setter.set(carrier, DEBUGG_AI_TRACE_HEADER_NAME, debugg_ai_span.to_traceparent())

        if debugg_ai_span.containing_transaction:
            baggage = debugg_ai_span.containing_transaction.get_baggage()
            if baggage:
                baggage_data = baggage.serialize()
                if baggage_data:
                    setter.set(carrier, BAGGAGE_HEADER_NAME, baggage_data)

    @property
    def fields(self):
        # type: () -> Set[str]
        return {DEBUGG_AI_TRACE_HEADER_NAME, BAGGAGE_HEADER_NAME}
