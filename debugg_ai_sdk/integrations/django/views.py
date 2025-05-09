import functools

import debugg_ai_sdk
from debugg_ai_sdk.consts import OP

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


try:
    from asyncio import iscoroutinefunction
except ImportError:
    iscoroutinefunction = None  # type: ignore


try:
    from debugg_ai_sdk.integrations.django.asgi import wrap_async_view
except (ImportError, SyntaxError):
    wrap_async_view = None  # type: ignore


def patch_views():
    # type: () -> None

    from django.core.handlers.base import BaseHandler
    from django.template.response import SimpleTemplateResponse
    from debugg_ai_sdk.integrations.django import DjangoIntegration

    old_make_view_atomic = BaseHandler.make_view_atomic
    old_render = SimpleTemplateResponse.render

    def debugg_ai_patched_render(self):
        # type: (SimpleTemplateResponse) -> Any
        with debugg_ai_sdk.start_span(
            op=OP.VIEW_RESPONSE_RENDER,
            name="serialize response",
            origin=DjangoIntegration.origin,
        ):
            return old_render(self)

    @functools.wraps(old_make_view_atomic)
    def debugg_ai_patched_make_view_atomic(self, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        callback = old_make_view_atomic(self, *args, **kwargs)

        # XXX: The wrapper function is created for every request. Find more
        # efficient way to wrap views (or build a cache?)

        integration = debugg_ai_sdk.get_client().get_integration(DjangoIntegration)
        if integration is not None and integration.middleware_spans:
            is_async_view = (
                iscoroutinefunction is not None
                and wrap_async_view is not None
                and iscoroutinefunction(callback)
            )
            if is_async_view:
                debugg_ai_wrapped_callback = wrap_async_view(callback)
            else:
                debugg_ai_wrapped_callback = _wrap_sync_view(callback)

        else:
            debugg_ai_wrapped_callback = callback

        return debugg_ai_wrapped_callback

    SimpleTemplateResponse.render = debugg_ai_patched_render
    BaseHandler.make_view_atomic = debugg_ai_patched_make_view_atomic


def _wrap_sync_view(callback):
    # type: (Any) -> Any
    from debugg_ai_sdk.integrations.django import DjangoIntegration

    @functools.wraps(callback)
    def debugg_ai_wrapped_callback(request, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        current_scope = debugg_ai_sdk.get_current_scope()
        if current_scope.transaction is not None:
            current_scope.transaction.update_active_thread()

        debugg_ai_scope = debugg_ai_sdk.get_isolation_scope()
        # set the active thread id to the handler thread for sync views
        # this isn't necessary for async views since that runs on main
        if debugg_ai_scope.profile is not None:
            debugg_ai_scope.profile.update_active_thread_id()

        with debugg_ai_sdk.start_span(
            op=OP.VIEW_RENDER,
            name=request.resolver_match.view_name,
            origin=DjangoIntegration.origin,
        ):
            return callback(request, *args, **kwargs)

    return debugg_ai_wrapped_callback
