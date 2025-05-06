from debugg_ai_sdk.scope import Scope
from debugg_ai_sdk.transport import Transport, HttpTransport
from debugg_ai_sdk.client import Client

from debugg_ai_sdk.api import *  # noqa

from debugg_ai_sdk.consts import VERSION  # noqa

__all__ = [  # noqa
    "Hub",
    "Scope",
    "Client",
    "Transport",
    "HttpTransport",
    "integrations",
    # From debugg_ai_sdk.api
    "init",
    "add_attachment",
    "add_breadcrumb",
    "capture_event",
    "capture_exception",
    "capture_message",
    "configure_scope",
    "continue_trace",
    "flush",
    "get_baggage",
    "get_client",
    "get_global_scope",
    "get_isolation_scope",
    "get_current_scope",
    "get_current_span",
    "get_traceparent",
    "is_initialized",
    "isolation_scope",
    "last_event_id",
    "new_scope",
    "push_scope",
    "set_context",
    "set_extra",
    "set_level",
    "set_measurement",
    "set_tag",
    "set_tags",
    "set_user",
    "start_span",
    "start_transaction",
    "trace",
    "monitor",
    "logger",
]

# Initialize the debug support after everything is loaded
from debugg_ai_sdk.debug import init_debug_support

init_debug_support()
del init_debug_support

# circular imports
from debugg_ai_sdk.hub import Hub
