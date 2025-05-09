import debugg_ai_sdk
from debugg_ai_sdk.integrations import Integration, DidNotEnable
from debugg_ai_sdk.scope import add_global_event_processor
from debugg_ai_sdk.utils import walk_exception_chain, iter_stacks

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from debugg_ai_sdk._types import Event, Hint

try:
    import executing
except ImportError:
    raise DidNotEnable("executing is not installed")


class ExecutingIntegration(Integration):
    identifier = "executing"

    @staticmethod
    def setup_once():
        # type: () -> None

        @add_global_event_processor
        def add_executing_info(event, hint):
            # type: (Event, Optional[Hint]) -> Optional[Event]
            if debugg_ai_sdk.get_client().get_integration(ExecutingIntegration) is None:
                return event

            if hint is None:
                return event

            exc_info = hint.get("exc_info", None)

            if exc_info is None:
                return event

            exception = event.get("exception", None)

            if exception is None:
                return event

            values = exception.get("values", None)

            if values is None:
                return event

            for exception, (_exc_type, _exc_value, exc_tb) in zip(
                reversed(values), walk_exception_chain(exc_info)
            ):
                debugg_ai_frames = [
                    frame
                    for frame in exception.get("stacktrace", {}).get("frames", [])
                    if frame.get("function")
                ]
                tbs = list(iter_stacks(exc_tb))
                if len(debugg_ai_frames) != len(tbs):
                    continue

                for debugg_ai_frame, tb in zip(debugg_ai_frames, tbs):
                    frame = tb.tb_frame
                    source = executing.Source.for_frame(frame)
                    debugg_ai_frame["function"] = source.code_qualname(frame.f_code)

            return event
