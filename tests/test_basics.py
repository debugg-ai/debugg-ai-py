import datetime
import importlib
import logging
import os
import sys
import time
from collections import Counter

import pytest
from debugg_ai_sdk.client import Client
from debugg_ai_sdk.utils import datetime_from_isoformat

import debugg_ai_sdk
import debugg_ai_sdk.scope
from debugg_ai_sdk import (
    get_client,
    push_scope,
    capture_event,
    capture_exception,
    capture_message,
    start_transaction,
    last_event_id,
    add_breadcrumb,
    isolation_scope,
    new_scope,
    Hub,
)
from debugg_ai_sdk.integrations import (
    _AUTO_ENABLING_INTEGRATIONS,
    _DEFAULT_INTEGRATIONS,
    DidNotEnable,
    Integration,
    setup_integrations,
)
from debugg_ai_sdk.integrations.logging import LoggingIntegration
from debugg_ai_sdk.integrations.stdlib import StdlibIntegration
from debugg_ai_sdk.scope import add_global_event_processor
from debugg_ai_sdk.utils import get_sdk_name, reraise
from debugg_ai_sdk.tracing_utils import has_tracing_enabled


class NoOpIntegration(Integration):
    """
    A simple no-op integration for testing purposes.
    """

    identifier = "noop"

    @staticmethod
    def setup_once():  # type: () -> None
        pass

    def __eq__(self, __value):  # type: (object) -> bool
        """
        All instances of NoOpIntegration should be considered equal to each other.
        """
        return type(__value) == type(self)


def test_processors(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()

    def error_processor(event, exc_info):
        event["exception"]["values"][0]["value"] += " whatever"
        return event

    debugg_ai_sdk.get_isolation_scope().add_error_processor(error_processor, ValueError)

    try:
        raise ValueError("aha!")
    except Exception:
        capture_exception()

    (event,) = events

    assert event["exception"]["values"][0]["value"] == "aha! whatever"


class ModuleImportErrorSimulator:
    def __init__(self, modules, error_cls=DidNotEnable):
        self.modules = modules
        self.error_cls = error_cls
        for sys_module in list(sys.modules.keys()):
            if any(sys_module.startswith(module) for module in modules):
                del sys.modules[sys_module]

    def find_spec(self, fullname, _path, _target=None):
        if fullname in self.modules:
            raise self.error_cls("Test import failure for %s" % fullname)

    def __enter__(self):
        # WARNING: We need to be first to avoid pytest messing with local imports
        sys.meta_path.insert(0, self)

    def __exit__(self, *_args):
        sys.meta_path.remove(self)


def test_auto_enabling_integrations_catches_import_error(debugg_ai_init, caplog):
    caplog.set_level(logging.DEBUG)

    with ModuleImportErrorSimulator(
        [i.rsplit(".", 1)[0] for i in _AUTO_ENABLING_INTEGRATIONS]
    ):
        debugg_ai_init(auto_enabling_integrations=True, debug=True)

    for import_string in _AUTO_ENABLING_INTEGRATIONS:
        assert any(
            record.message.startswith(
                "Did not import default integration {}:".format(import_string)
            )
            for record in caplog.records
        ), "Problem with checking auto enabling {}".format(import_string)


def test_generic_mechanism(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()

    try:
        raise ValueError("aha!")
    except Exception:
        capture_exception()

    (event,) = events
    assert event["exception"]["values"][0]["mechanism"]["type"] == "generic"
    assert event["exception"]["values"][0]["mechanism"]["handled"]


def test_option_before_send(debugg_ai_init, capture_events):
    def before_send(event, hint):
        event["extra"] = {"before_send_called": True}
        return event

    def do_this():
        try:
            raise ValueError("aha!")
        except Exception:
            capture_exception()

    debugg_ai_init(before_send=before_send)
    events = capture_events()

    do_this()

    (event,) = events
    assert event["extra"] == {"before_send_called": True}


def test_option_before_send_discard(debugg_ai_init, capture_events):
    def before_send_discard(event, hint):
        return None

    def do_this():
        try:
            raise ValueError("aha!")
        except Exception:
            capture_exception()

    debugg_ai_init(before_send=before_send_discard)
    events = capture_events()

    do_this()

    assert len(events) == 0


def test_option_before_send_transaction(debugg_ai_init, capture_events):
    def before_send_transaction(event, hint):
        assert event["type"] == "transaction"
        event["extra"] = {"before_send_transaction_called": True}
        return event

    debugg_ai_init(
        before_send_transaction=before_send_transaction,
        traces_sample_rate=1.0,
    )
    events = capture_events()
    transaction = start_transaction(name="foo")
    transaction.finish()

    (event,) = events
    assert event["transaction"] == "foo"
    assert event["extra"] == {"before_send_transaction_called": True}


def test_option_before_send_transaction_discard(debugg_ai_init, capture_events):
    def before_send_transaction_discard(event, hint):
        return None

    debugg_ai_init(
        before_send_transaction=before_send_transaction_discard,
        traces_sample_rate=1.0,
    )
    events = capture_events()
    transaction = start_transaction(name="foo")
    transaction.finish()

    assert len(events) == 0


def test_option_before_breadcrumb(debugg_ai_init, capture_events, monkeypatch):
    drop_events = False
    drop_breadcrumbs = False
    reports = []

    def record_lost_event(reason, data_category=None, item=None):
        reports.append((reason, data_category))

    def before_send(event, hint):
        assert isinstance(hint["exc_info"][1], ValueError)
        if not drop_events:
            event["extra"] = {"foo": "bar"}
            return event

    def before_breadcrumb(crumb, hint):
        assert hint == {"foo": 42}
        if not drop_breadcrumbs:
            crumb["data"] = {"foo": "bar"}
            return crumb

    debugg_ai_init(before_send=before_send, before_breadcrumb=before_breadcrumb)
    events = capture_events()

    monkeypatch.setattr(
        debugg_ai_sdk.get_client().transport, "record_lost_event", record_lost_event
    )

    def do_this():
        add_breadcrumb(message="Hello", hint={"foo": 42})
        try:
            raise ValueError("aha!")
        except Exception:
            capture_exception()

    do_this()
    drop_breadcrumbs = True
    do_this()
    assert not reports
    drop_events = True
    do_this()
    assert reports == [("before_send", "error")]

    normal, no_crumbs = events

    assert normal["exception"]["values"][0]["type"] == "ValueError"
    (crumb,) = normal["breadcrumbs"]["values"]
    assert "timestamp" in crumb
    assert crumb["message"] == "Hello"
    assert crumb["data"] == {"foo": "bar"}
    assert crumb["type"] == "default"


@pytest.mark.parametrize(
    "enable_tracing, traces_sample_rate, tracing_enabled, updated_traces_sample_rate",
    [
        (None, None, False, None),
        (False, 0.0, False, 0.0),
        (False, 1.0, False, 1.0),
        (None, 1.0, True, 1.0),
        (True, 1.0, True, 1.0),
        (None, 0.0, True, 0.0),  # We use this as - it's configured but turned off
        (True, 0.0, True, 0.0),  # We use this as - it's configured but turned off
        (True, None, True, 1.0),
    ],
)
def test_option_enable_tracing(
    debugg_ai_init,
    enable_tracing,
    traces_sample_rate,
    tracing_enabled,
    updated_traces_sample_rate,
):
    debugg_ai_init(enable_tracing=enable_tracing, traces_sample_rate=traces_sample_rate)
    options = debugg_ai_sdk.get_client().options
    assert has_tracing_enabled(options) is tracing_enabled
    assert options["traces_sample_rate"] == updated_traces_sample_rate


def test_breadcrumb_arguments(debugg_ai_init, capture_events):
    assert_hint = {"bar": 42}

    def before_breadcrumb(crumb, hint):
        assert crumb["foo"] == 42
        assert hint == assert_hint

    debugg_ai_init(before_breadcrumb=before_breadcrumb)

    add_breadcrumb(foo=42, hint=dict(bar=42))
    add_breadcrumb(dict(foo=42), dict(bar=42))
    add_breadcrumb(dict(foo=42), hint=dict(bar=42))
    add_breadcrumb(crumb=dict(foo=42), hint=dict(bar=42))

    assert_hint.clear()
    add_breadcrumb(foo=42)
    add_breadcrumb(crumb=dict(foo=42))


def test_push_scope(debugg_ai_init, capture_events, suppress_deprecation_warnings):
    debugg_ai_init()
    events = capture_events()

    with push_scope() as scope:
        scope.level = "warning"
        try:
            1 / 0
        except Exception as e:
            capture_exception(e)

    (event,) = events

    assert event["level"] == "warning"
    assert "exception" in event


def test_push_scope_null_client(
    debugg_ai_init, capture_events, suppress_deprecation_warnings
):
    """
    This test can be removed when we remove push_scope and the Hub from the SDK.
    """
    debugg_ai_init()
    events = capture_events()

    Hub.current.bind_client(None)

    with push_scope() as scope:
        scope.level = "warning"
        try:
            1 / 0
        except Exception as e:
            capture_exception(e)

    assert len(events) == 0


@pytest.mark.skip(
    reason="This test is not valid anymore, because push_scope just returns the isolation scope. This test should be removed once the Hub is removed"
)
@pytest.mark.parametrize("null_client", (True, False))
def test_push_scope_callback(debugg_ai_init, null_client, capture_events):
    """
    This test can be removed when we remove push_scope and the Hub from the SDK.
    """
    debugg_ai_init()

    if null_client:
        Hub.current.bind_client(None)

    outer_scope = Hub.current.scope

    calls = []

    @push_scope
    def _(scope):
        assert scope is Hub.current.scope
        assert scope is not outer_scope
        calls.append(1)

    # push_scope always needs to execute the callback regardless of
    # client state, because that actually runs usercode in it, not
    # just scope config code
    assert calls == [1]

    # Assert scope gets popped correctly
    assert Hub.current.scope is outer_scope


def test_breadcrumbs(debugg_ai_init, capture_events):
    debugg_ai_init(max_breadcrumbs=10)
    events = capture_events()

    for i in range(20):
        add_breadcrumb(
            category="auth", message="Authenticated user %s" % i, level="info"
        )

    capture_exception(ValueError())
    (event,) = events

    assert len(event["breadcrumbs"]["values"]) == 10
    assert "user 10" in event["breadcrumbs"]["values"][0]["message"]
    assert "user 19" in event["breadcrumbs"]["values"][-1]["message"]

    del events[:]

    for i in range(2):
        add_breadcrumb(
            category="auth", message="Authenticated user %s" % i, level="info"
        )

    debugg_ai_sdk.get_isolation_scope().clear()

    capture_exception(ValueError())
    (event,) = events
    assert len(event["breadcrumbs"]["values"]) == 0


def test_breadcrumb_ordering(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)

    timestamps = [
        now - datetime.timedelta(days=10),
        now - datetime.timedelta(days=8),
        now - datetime.timedelta(days=12),
    ]

    for timestamp in timestamps:
        add_breadcrumb(
            message="Authenticated at %s" % timestamp,
            category="auth",
            level="info",
            timestamp=timestamp,
        )

    capture_exception(ValueError())
    (event,) = events

    assert len(event["breadcrumbs"]["values"]) == len(timestamps)
    timestamps_from_event = [
        datetime_from_isoformat(x["timestamp"]) for x in event["breadcrumbs"]["values"]
    ]
    assert timestamps_from_event == sorted(timestamps)


def test_breadcrumb_ordering_different_types(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()
    now = datetime.datetime.now(datetime.timezone.utc)

    timestamps = [
        now - datetime.timedelta(days=10),
        now - datetime.timedelta(days=8),
        now.replace(microsecond=0) - datetime.timedelta(days=12),
        now - datetime.timedelta(days=9),
        now - datetime.timedelta(days=13),
        now.replace(microsecond=0) - datetime.timedelta(days=11),
    ]

    breadcrumb_timestamps = [
        timestamps[0],
        timestamps[1].isoformat(),
        datetime.datetime.strftime(timestamps[2], "%Y-%m-%dT%H:%M:%S") + "Z",
        datetime.datetime.strftime(timestamps[3], "%Y-%m-%dT%H:%M:%S.%f") + "+00:00",
        datetime.datetime.strftime(timestamps[4], "%Y-%m-%dT%H:%M:%S.%f") + "+0000",
        datetime.datetime.strftime(timestamps[5], "%Y-%m-%dT%H:%M:%S.%f") + "-0000",
    ]

    for i, timestamp in enumerate(timestamps):
        add_breadcrumb(
            message="Authenticated at %s" % timestamp,
            category="auth",
            level="info",
            timestamp=breadcrumb_timestamps[i],
        )

    capture_exception(ValueError())
    (event,) = events

    assert len(event["breadcrumbs"]["values"]) == len(timestamps)
    timestamps_from_event = [
        datetime_from_isoformat(x["timestamp"]) for x in event["breadcrumbs"]["values"]
    ]
    assert timestamps_from_event == sorted(timestamps)


def test_attachments(debugg_ai_init, capture_envelopes):
    debugg_ai_init()
    envelopes = capture_envelopes()

    this_file = os.path.abspath(__file__.rstrip("c"))

    scope = debugg_ai_sdk.get_isolation_scope()
    scope.add_attachment(bytes=b"Hello World!", filename="message.txt")
    scope.add_attachment(path=this_file)

    capture_exception(ValueError())

    (envelope,) = envelopes

    assert len(envelope.items) == 3
    assert envelope.get_event()["exception"] is not None

    attachments = [x for x in envelope.items if x.type == "attachment"]
    (message, pyfile) = attachments

    assert message.headers["filename"] == "message.txt"
    assert message.headers["type"] == "attachment"
    assert message.headers["content_type"] == "text/plain"
    assert message.payload.bytes == message.payload.get_bytes() == b"Hello World!"

    assert pyfile.headers["filename"] == os.path.basename(this_file)
    assert pyfile.headers["type"] == "attachment"
    assert pyfile.headers["content_type"].startswith("text/")
    assert pyfile.payload.bytes is None
    with open(this_file, "rb") as f:
        assert pyfile.payload.get_bytes() == f.read()


@pytest.mark.tests_internal_exceptions
def test_attachments_graceful_failure(
    debugg_ai_init, capture_envelopes, internal_exceptions
):
    debugg_ai_init()
    envelopes = capture_envelopes()

    debugg_ai_sdk.get_isolation_scope().add_attachment(path="non_existent")
    capture_exception(ValueError())

    (envelope,) = envelopes
    assert len(envelope.items) == 2
    assert envelope.items[1].payload.get_bytes() == b""


def test_integration_scoping(debugg_ai_init, capture_events):
    logger = logging.getLogger("test_basics")

    # This client uses the logging integration
    logging_integration = LoggingIntegration(event_level=logging.WARNING)
    debugg_ai_init(default_integrations=False, integrations=[logging_integration])
    events = capture_events()
    logger.warning("This is a warning")
    assert len(events) == 1

    # This client does not
    debugg_ai_init(default_integrations=False)
    events = capture_events()
    logger.warning("This is not a warning")
    assert not events


default_integrations = [
    getattr(
        importlib.import_module(integration.rsplit(".", 1)[0]),
        integration.rsplit(".", 1)[1],
    )
    for integration in _DEFAULT_INTEGRATIONS
]


@pytest.mark.forked
@pytest.mark.parametrize(
    "provided_integrations,default_integrations,disabled_integrations,expected_integrations",
    [
        ([], False, None, set()),
        ([], False, [], set()),
        ([LoggingIntegration()], False, None, {LoggingIntegration}),
        ([], True, None, set(default_integrations)),
        (
            [],
            True,
            [LoggingIntegration(), StdlibIntegration],
            set(default_integrations) - {LoggingIntegration, StdlibIntegration},
        ),
    ],
)
def test_integrations(
    debugg_ai_init,
    provided_integrations,
    default_integrations,
    disabled_integrations,
    expected_integrations,
    reset_integrations,
):
    debugg_ai_init(
        integrations=provided_integrations,
        default_integrations=default_integrations,
        disabled_integrations=disabled_integrations,
        auto_enabling_integrations=False,
        debug=True,
    )
    assert {
        type(integration) for integration in get_client().integrations.values()
    } == expected_integrations


@pytest.mark.skip(
    reason="This test is not valid anymore, because with the new Scopes calling bind_client on the Hub sets the client on the global scope. This test should be removed once the Hub is removed"
)
def test_client_initialized_within_scope(debugg_ai_init, caplog):
    """
    This test can be removed when we remove push_scope and the Hub from the SDK.
    """
    caplog.set_level(logging.WARNING)

    debugg_ai_init()

    with push_scope():
        Hub.current.bind_client(Client())

    (record,) = (x for x in caplog.records if x.levelname == "WARNING")

    assert record.msg.startswith("init() called inside of pushed scope.")


@pytest.mark.skip(
    reason="This test is not valid anymore, because with the new Scopes the push_scope just returns the isolation scope. This test should be removed once the Hub is removed"
)
def test_scope_leaks_cleaned_up(debugg_ai_init, caplog):
    """
    This test can be removed when we remove push_scope and the Hub from the SDK.
    """
    caplog.set_level(logging.WARNING)

    debugg_ai_init()

    old_stack = list(Hub.current._stack)

    with push_scope():
        push_scope()

    assert Hub.current._stack == old_stack

    (record,) = (x for x in caplog.records if x.levelname == "WARNING")

    assert record.message.startswith("Leaked 1 scopes:")


@pytest.mark.skip(
    reason="This test is not valid anymore, because with the new Scopes there is not pushing and popping of scopes. This test should be removed once the Hub is removed"
)
def test_scope_popped_too_soon(debugg_ai_init, caplog):
    """
    This test can be removed when we remove push_scope and the Hub from the SDK.
    """
    caplog.set_level(logging.ERROR)

    debugg_ai_init()

    old_stack = list(Hub.current._stack)

    with push_scope():
        Hub.current.pop_scope_unsafe()

    assert Hub.current._stack == old_stack

    (record,) = (x for x in caplog.records if x.levelname == "ERROR")

    assert record.message == ("Scope popped too soon. Popped 1 scopes too many.")


def test_scope_event_processor_order(debugg_ai_init, capture_events):
    def before_send(event, hint):
        event["message"] += "baz"
        return event

    debugg_ai_init(debug=True, before_send=before_send)
    events = capture_events()

    with new_scope() as scope:

        @scope.add_event_processor
        def foo(event, hint):
            event["message"] += "foo"
            return event

        with new_scope() as scope:

            @scope.add_event_processor
            def bar(event, hint):
                event["message"] += "bar"
                return event

            capture_message("hi")

    (event,) = events

    assert event["message"] == "hifoobarbaz"


def test_capture_event_with_scope_kwargs(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()
    capture_event({}, level="info", extras={"foo": "bar"})
    (event,) = events
    assert event["level"] == "info"
    assert event["extra"]["foo"] == "bar"


def test_dedupe_event_processor_drop_records_client_report(
    debugg_ai_init, capture_events, capture_record_lost_event_calls
):
    """
    DedupeIntegration internally has an event_processor that filters duplicate exceptions.
    We want a duplicate exception to be captured only once and the drop being recorded as
    a client report.
    """
    debugg_ai_init()
    events = capture_events()
    record_lost_event_calls = capture_record_lost_event_calls()

    try:
        raise ValueError("aha!")
    except Exception:
        try:
            capture_exception()
            reraise(*sys.exc_info())
        except Exception:
            capture_exception()

    (event,) = events
    (lost_event_call,) = record_lost_event_calls

    assert event["level"] == "error"
    assert "exception" in event
    assert lost_event_call == ("event_processor", "error", None, 1)


def test_dedupe_doesnt_take_into_account_dropped_exception(debugg_ai_init, capture_events):
    # Two exceptions happen one after another. The first one is dropped in the
    # user's before_send. The second one isn't.
    # Originally, DedupeIntegration would drop the second exception. This test
    # is making sure that that is no longer the case -- i.e., DedupeIntegration
    # doesn't consider exceptions dropped in before_send.
    count = 0

    def before_send(event, hint):
        nonlocal count
        count += 1
        if count == 1:
            return None
        return event

    debugg_ai_init(before_send=before_send)
    events = capture_events()

    exc = ValueError("aha!")
    for _ in range(2):
        # The first ValueError will be dropped by before_send. The second
        # ValueError will be accepted by before_send, and should be sent to
        # DebuggAI.
        try:
            raise exc
        except Exception:
            capture_exception()

    assert len(events) == 1


def test_event_processor_drop_records_client_report(
    debugg_ai_init, capture_events, capture_record_lost_event_calls
):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()
    record_lost_event_calls = capture_record_lost_event_calls()

    # Ensure full idempotency by restoring the original global event processors list object, not just a copy.
    old_processors = debugg_ai_sdk.scope.global_event_processors

    try:
        debugg_ai_sdk.scope.global_event_processors = (
            debugg_ai_sdk.scope.global_event_processors.copy()
        )

        @add_global_event_processor
        def foo(event, hint):
            return None

        capture_message("dropped")

        with start_transaction(name="dropped"):
            pass

        assert len(events) == 0

        # Using Counter because order of record_lost_event calls does not matter
        assert Counter(record_lost_event_calls) == Counter(
            [
                ("event_processor", "error", None, 1),
                ("event_processor", "transaction", None, 1),
                ("event_processor", "span", None, 1),
            ]
        )

    finally:
        debugg_ai_sdk.scope.global_event_processors = old_processors


@pytest.mark.parametrize(
    "installed_integrations, expected_name",
    [
        # integrations with own name
        (["django"], "debugg-ai.python.django"),
        (["flask"], "debugg-ai.python.flask"),
        (["fastapi"], "debugg-ai.python.fastapi"),
        (["bottle"], "debugg-ai.python.bottle"),
        (["falcon"], "debugg-ai.python.falcon"),
        (["quart"], "debugg-ai.python.quart"),
        (["sanic"], "debugg-ai.python.sanic"),
        (["starlette"], "debugg-ai.python.starlette"),
        (["starlite"], "debugg-ai.python.starlite"),
        (["litestar"], "debugg-ai.python.litestar"),
        (["chalice"], "debugg-ai.python.chalice"),
        (["serverless"], "debugg-ai.python.serverless"),
        (["pyramid"], "debugg-ai.python.pyramid"),
        (["tornado"], "debugg-ai.python.tornado"),
        (["aiohttp"], "debugg-ai.python.aiohttp"),
        (["aws_lambda"], "debugg-ai.python.aws_lambda"),
        (["gcp"], "debugg-ai.python.gcp"),
        (["beam"], "debugg-ai.python.beam"),
        (["asgi"], "debugg-ai.python.asgi"),
        (["wsgi"], "debugg-ai.python.wsgi"),
        # integrations without name
        (["argv"], "debugg-ai.python"),
        (["atexit"], "debugg-ai.python"),
        (["boto3"], "debugg-ai.python"),
        (["celery"], "debugg-ai.python"),
        (["dedupe"], "debugg-ai.python"),
        (["excepthook"], "debugg-ai.python"),
        (["executing"], "debugg-ai.python"),
        (["modules"], "debugg-ai.python"),
        (["pure_eval"], "debugg-ai.python"),
        (["redis"], "debugg-ai.python"),
        (["rq"], "debugg-ai.python"),
        (["sqlalchemy"], "debugg-ai.python"),
        (["stdlib"], "debugg-ai.python"),
        (["threading"], "debugg-ai.python"),
        (["trytond"], "debugg-ai.python"),
        (["logging"], "debugg-ai.python"),
        (["gnu_backtrace"], "debugg-ai.python"),
        (["httpx"], "debugg-ai.python"),
        # precedence of frameworks
        (["flask", "django", "celery"], "debugg-ai.python.django"),
        (["fastapi", "flask", "redis"], "debugg-ai.python.flask"),
        (["bottle", "fastapi", "httpx"], "debugg-ai.python.fastapi"),
        (["falcon", "bottle", "logging"], "debugg-ai.python.bottle"),
        (["quart", "falcon", "gnu_backtrace"], "debugg-ai.python.falcon"),
        (["sanic", "quart", "sqlalchemy"], "debugg-ai.python.quart"),
        (["starlette", "sanic", "rq"], "debugg-ai.python.sanic"),
        (["chalice", "starlette", "modules"], "debugg-ai.python.starlette"),
        (["chalice", "starlite", "modules"], "debugg-ai.python.starlite"),
        (["chalice", "litestar", "modules"], "debugg-ai.python.litestar"),
        (["serverless", "chalice", "pure_eval"], "debugg-ai.python.chalice"),
        (["pyramid", "serverless", "modules"], "debugg-ai.python.serverless"),
        (["tornado", "pyramid", "executing"], "debugg-ai.python.pyramid"),
        (["aiohttp", "tornado", "dedupe"], "debugg-ai.python.tornado"),
        (["aws_lambda", "aiohttp", "boto3"], "debugg-ai.python.aiohttp"),
        (["gcp", "aws_lambda", "atexit"], "debugg-ai.python.aws_lambda"),
        (["beam", "gcp", "argv"], "debugg-ai.python.gcp"),
        (["asgi", "beam", "stdtlib"], "debugg-ai.python.beam"),
        (["wsgi", "asgi", "boto3"], "debugg-ai.python.asgi"),
        (["wsgi", "celery", "redis"], "debugg-ai.python.wsgi"),
    ],
)
def test_get_sdk_name(installed_integrations, expected_name):
    assert get_sdk_name(installed_integrations) == expected_name


def _hello_world(word):
    return "Hello, {}".format(word)


def test_functions_to_trace(debugg_ai_init, capture_events):
    functions_to_trace = [
        {"qualified_name": "tests.test_basics._hello_world"},
        {"qualified_name": "time.sleep"},
    ]

    debugg_ai_init(
        traces_sample_rate=1.0,
        functions_to_trace=functions_to_trace,
    )

    events = capture_events()

    with start_transaction(name="something"):
        time.sleep(0)

        for word in ["World", "You"]:
            _hello_world(word)

    assert len(events) == 1

    (event,) = events

    assert len(event["spans"]) == 3
    assert event["spans"][0]["description"] == "time.sleep"
    assert event["spans"][1]["description"] == "tests.test_basics._hello_world"
    assert event["spans"][2]["description"] == "tests.test_basics._hello_world"


class WorldGreeter:
    def __init__(self, word):
        self.word = word

    def greet(self, new_word=None):
        return "Hello, {}".format(new_word if new_word else self.word)


def test_functions_to_trace_with_class(debugg_ai_init, capture_events):
    functions_to_trace = [
        {"qualified_name": "tests.test_basics.WorldGreeter.greet"},
    ]

    debugg_ai_init(
        traces_sample_rate=1.0,
        functions_to_trace=functions_to_trace,
    )

    events = capture_events()

    with start_transaction(name="something"):
        wg = WorldGreeter("World")
        wg.greet()
        wg.greet("You")

    assert len(events) == 1

    (event,) = events

    assert len(event["spans"]) == 2
    assert event["spans"][0]["description"] == "tests.test_basics.WorldGreeter.greet"
    assert event["spans"][1]["description"] == "tests.test_basics.WorldGreeter.greet"


def test_multiple_setup_integrations_calls():
    first_call_return = setup_integrations([NoOpIntegration()], with_defaults=False)
    assert first_call_return == {NoOpIntegration.identifier: NoOpIntegration()}

    second_call_return = setup_integrations([NoOpIntegration()], with_defaults=False)
    assert second_call_return == {NoOpIntegration.identifier: NoOpIntegration()}


class TracingTestClass:
    @staticmethod
    def static(arg):
        return arg

    @classmethod
    def class_(cls, arg):
        return cls, arg


# We need to fork here because the test modifies tests.test_basics.TracingTestClass
@pytest.mark.forked
def test_staticmethod_class_tracing(debugg_ai_init, capture_events):
    debugg_ai_init(
        debug=True,
        traces_sample_rate=1.0,
        functions_to_trace=[
            {"qualified_name": "tests.test_basics.TracingTestClass.static"}
        ],
    )

    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="test"):
        assert TracingTestClass.static(1) == 1

    (event,) = events
    assert event["type"] == "transaction"
    assert event["transaction"] == "test"

    (span,) = event["spans"]
    assert span["description"] == "tests.test_basics.TracingTestClass.static"


# We need to fork here because the test modifies tests.test_basics.TracingTestClass
@pytest.mark.forked
def test_staticmethod_instance_tracing(debugg_ai_init, capture_events):
    debugg_ai_init(
        debug=True,
        traces_sample_rate=1.0,
        functions_to_trace=[
            {"qualified_name": "tests.test_basics.TracingTestClass.static"}
        ],
    )

    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="test"):
        assert TracingTestClass().static(1) == 1

    (event,) = events
    assert event["type"] == "transaction"
    assert event["transaction"] == "test"

    (span,) = event["spans"]
    assert span["description"] == "tests.test_basics.TracingTestClass.static"


# We need to fork here because the test modifies tests.test_basics.TracingTestClass
@pytest.mark.forked
def test_classmethod_class_tracing(debugg_ai_init, capture_events):
    debugg_ai_init(
        debug=True,
        traces_sample_rate=1.0,
        functions_to_trace=[
            {"qualified_name": "tests.test_basics.TracingTestClass.class_"}
        ],
    )

    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="test"):
        assert TracingTestClass.class_(1) == (TracingTestClass, 1)

    (event,) = events
    assert event["type"] == "transaction"
    assert event["transaction"] == "test"

    (span,) = event["spans"]
    assert span["description"] == "tests.test_basics.TracingTestClass.class_"


# We need to fork here because the test modifies tests.test_basics.TracingTestClass
@pytest.mark.forked
def test_classmethod_instance_tracing(debugg_ai_init, capture_events):
    debugg_ai_init(
        debug=True,
        traces_sample_rate=1.0,
        functions_to_trace=[
            {"qualified_name": "tests.test_basics.TracingTestClass.class_"}
        ],
    )

    events = capture_events()

    with debugg_ai_sdk.start_transaction(name="test"):
        assert TracingTestClass().class_(1) == (TracingTestClass, 1)

    (event,) = events
    assert event["type"] == "transaction"
    assert event["transaction"] == "test"

    (span,) = event["spans"]
    assert span["description"] == "tests.test_basics.TracingTestClass.class_"


def test_last_event_id(debugg_ai_init):
    debugg_ai_init(enable_tracing=True)

    assert last_event_id() is None

    capture_exception(Exception("test"))

    assert last_event_id() is not None


def test_last_event_id_transaction(debugg_ai_init):
    debugg_ai_init(enable_tracing=True)

    assert last_event_id() is None

    with start_transaction(name="test"):
        pass

    assert last_event_id() is None, "Transaction should not set last_event_id"


def test_last_event_id_scope(debugg_ai_init):
    debugg_ai_init(enable_tracing=True)

    # Should not crash
    with isolation_scope() as scope:
        assert scope.last_event_id() is None


def test_hub_constructor_deprecation_warning():
    with pytest.warns(debugg_ai_sdk.hub.DebuggAIHubDeprecationWarning):
        Hub()


def test_hub_current_deprecation_warning():
    with pytest.warns(debugg_ai_sdk.hub.DebuggAIHubDeprecationWarning) as warning_records:
        Hub.current

    # Make sure we only issue one deprecation warning
    assert len(warning_records) == 1


def test_hub_main_deprecation_warnings():
    with pytest.warns(debugg_ai_sdk.hub.DebuggAIHubDeprecationWarning):
        Hub.main


@pytest.mark.skipif(sys.version_info < (3, 11), reason="add_note() not supported")
def test_notes(debugg_ai_init, capture_events):
    debugg_ai_init()
    events = capture_events()
    try:
        e = ValueError("aha!")
        e.add_note("Test 123")
        e.add_note("another note")
        raise e
    except Exception:
        capture_exception()

    (event,) = events

    assert event["exception"]["values"][0]["value"] == "aha!\nTest 123\nanother note"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="add_note() not supported")
def test_notes_safe_str(debugg_ai_init, capture_events):
    class Note2:
        def __repr__(self):
            raise TypeError

        def __str__(self):
            raise TypeError

    debugg_ai_init()
    events = capture_events()
    try:
        e = ValueError("aha!")
        e.add_note("note 1")
        e.__notes__.append(Note2())  # type: ignore
        e.add_note("note 3")
        e.__notes__.append(2)  # type: ignore
        raise e
    except Exception:
        capture_exception()

    (event,) = events

    assert event["exception"]["values"][0]["value"] == "aha!\nnote 1\nnote 3"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="this test appears to cause a segfault on Python < 3.11",
)
def test_stacktrace_big_recursion(debugg_ai_init, capture_events):
    """
    Ensure that if the recursion limit is increased, the full stacktrace is not captured,
    as it would take too long to process the entire stack trace.
    Also, ensure that the capturing does not take too long.
    """
    debugg_ai_init()
    events = capture_events()

    def recurse():
        recurse()

    old_recursion_limit = sys.getrecursionlimit()

    try:
        sys.setrecursionlimit(100_000)
        recurse()
    except RecursionError as e:
        capture_start_time = time.perf_counter_ns()
        debugg_ai_sdk.capture_exception(e)
        capture_end_time = time.perf_counter_ns()
    finally:
        sys.setrecursionlimit(old_recursion_limit)

    (event,) = events

    assert event["exception"]["values"][0]["stacktrace"] is None
    assert event["_meta"]["exception"] == {
        "values": {"0": {"stacktrace": {"": {"rem": [["!config", "x"]]}}}}
    }

    # On my machine, it takes about 100-200ms to capture the exception,
    # so this limit should be generous enough.
    assert (
        capture_end_time - capture_start_time < 10**9 * 2
    ), "stacktrace capture took too long, check that frame limit is set correctly"
