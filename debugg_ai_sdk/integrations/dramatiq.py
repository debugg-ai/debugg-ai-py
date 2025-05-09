import json

import debugg_ai_sdk
from debugg_ai_sdk.integrations import Integration
from debugg_ai_sdk.integrations._wsgi_common import request_body_within_bounds
from debugg_ai_sdk.utils import (
    AnnotatedValue,
    capture_internal_exceptions,
    event_from_exception,
)

from dramatiq.broker import Broker  # type: ignore
from dramatiq.message import Message  # type: ignore
from dramatiq.middleware import Middleware, default_middleware  # type: ignore
from dramatiq.errors import Retry  # type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional, Union
    from debugg_ai_sdk._types import Event, Hint


class DramatiqIntegration(Integration):
    """
    Dramatiq integration for DebuggAI

    Please make sure that you call `debugg_ai_sdk.init` *before* initializing
    your broker, as it monkey patches `Broker.__init__`.

    This integration was originally developed and maintained
    by https://github.com/jacobsvante and later donated to the DebuggAI
    project.
    """

    identifier = "dramatiq"

    @staticmethod
    def setup_once():
        # type: () -> None
        _patch_dramatiq_broker()


def _patch_dramatiq_broker():
    # type: () -> None
    original_broker__init__ = Broker.__init__

    def debugg_ai_patched_broker__init__(self, *args, **kw):
        # type: (Broker, *Any, **Any) -> None
        integration = debugg_ai_sdk.get_client().get_integration(DramatiqIntegration)

        try:
            middleware = kw.pop("middleware")
        except KeyError:
            # Unfortunately Broker and StubBroker allows middleware to be
            # passed in as positional arguments, whilst RabbitmqBroker and
            # RedisBroker does not.
            if len(args) == 1:
                middleware = args[0]
                args = []  # type: ignore
            else:
                middleware = None

        if middleware is None:
            middleware = list(m() for m in default_middleware)
        else:
            middleware = list(middleware)

        if integration is not None:
            middleware = [m for m in middleware if not isinstance(m, DebuggAIMiddleware)]
            middleware.insert(0, DebuggAIMiddleware())

        kw["middleware"] = middleware
        original_broker__init__(self, *args, **kw)

    Broker.__init__ = debugg_ai_patched_broker__init__


class DebuggAIMiddleware(Middleware):  # type: ignore[misc]
    """
    A Dramatiq middleware that automatically captures and sends
    exceptions to DebuggAI.

    This is automatically added to every instantiated broker via the
    DramatiqIntegration.
    """

    def before_process_message(self, broker, message):
        # type: (Broker, Message) -> None
        integration = debugg_ai_sdk.get_client().get_integration(DramatiqIntegration)
        if integration is None:
            return

        message._scope_manager = debugg_ai_sdk.new_scope()
        message._scope_manager.__enter__()

        scope = debugg_ai_sdk.get_current_scope()
        scope.set_transaction_name(message.actor_name)
        scope.set_extra("dramatiq_message_id", message.message_id)
        scope.add_event_processor(_make_message_event_processor(message, integration))

    def after_process_message(self, broker, message, *, result=None, exception=None):
        # type: (Broker, Message, Any, Optional[Any], Optional[Exception]) -> None
        integration = debugg_ai_sdk.get_client().get_integration(DramatiqIntegration)
        if integration is None:
            return

        actor = broker.get_actor(message.actor_name)
        throws = message.options.get("throws") or actor.options.get("throws")

        try:
            if (
                exception is not None
                and not (throws and isinstance(exception, throws))
                and not isinstance(exception, Retry)
            ):
                event, hint = event_from_exception(
                    exception,
                    client_options=debugg_ai_sdk.get_client().options,
                    mechanism={
                        "type": DramatiqIntegration.identifier,
                        "handled": False,
                    },
                )
                debugg_ai_sdk.capture_event(event, hint=hint)
        finally:
            message._scope_manager.__exit__(None, None, None)


def _make_message_event_processor(message, integration):
    # type: (Message, DramatiqIntegration) -> Callable[[Event, Hint], Optional[Event]]

    def inner(event, hint):
        # type: (Event, Hint) -> Optional[Event]
        with capture_internal_exceptions():
            DramatiqMessageExtractor(message).extract_into_event(event)

        return event

    return inner


class DramatiqMessageExtractor:
    def __init__(self, message):
        # type: (Message) -> None
        self.message_data = dict(message.asdict())

    def content_length(self):
        # type: () -> int
        return len(json.dumps(self.message_data))

    def extract_into_event(self, event):
        # type: (Event) -> None
        client = debugg_ai_sdk.get_client()
        if not client.is_active():
            return

        contexts = event.setdefault("contexts", {})
        request_info = contexts.setdefault("dramatiq", {})
        request_info["type"] = "dramatiq"

        data = None  # type: Optional[Union[AnnotatedValue, Dict[str, Any]]]
        if not request_body_within_bounds(client, self.content_length()):
            data = AnnotatedValue.removed_because_over_size_limit()
        else:
            data = self.message_data

        request_info["data"] = data
