import json
import pytest
import logging

from io import BytesIO
from bottle import Bottle, debug as set_debug, abort, redirect, HTTPResponse
from debugg_ai_sdk import capture_message
from debugg_ai_sdk.integrations.bottle import BottleIntegration
from debugg_ai_sdk.serializer import MAX_DATABAG_BREADTH

from debugg_ai_sdk.integrations.logging import LoggingIntegration
from werkzeug.test import Client
from werkzeug.wrappers import Response

import debugg_ai_sdk.integrations.bottle as bottle_debugg-ai


@pytest.fixture(scope="function")
def app(debugg_ai_init):
    app = Bottle()

    @app.route("/message")
    def hi():
        capture_message("hi")
        return "ok"

    @app.route("/message/<message_id>")
    def hi_with_id(message_id):
        capture_message("hi")
        return "ok"

    @app.route("/message-named-route", name="hi")
    def named_hi():
        capture_message("hi")
        return "ok"

    yield app


@pytest.fixture
def get_client(app):
    def inner():
        return Client(app)

    return inner


def test_has_context(debugg_ai_init, app, capture_events, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])
    events = capture_events()

    client = get_client()
    response = client.get("/message")
    assert response[1] == "200 OK"

    (event,) = events
    assert event["message"] == "hi"
    assert "data" not in event["request"]
    assert event["request"]["url"] == "http://localhost/message"


@pytest.mark.parametrize(
    "url,transaction_style,expected_transaction,expected_source",
    [
        ("/message", "endpoint", "hi", "component"),
        ("/message", "url", "/message", "route"),
        ("/message/123456", "url", "/message/<message_id>", "route"),
        ("/message-named-route", "endpoint", "hi", "component"),
    ],
)
def test_transaction_style(
    debugg_ai_init,
    url,
    transaction_style,
    expected_transaction,
    expected_source,
    capture_events,
    get_client,
):
    debugg_ai_init(
        integrations=[
            bottle_debugg-ai.BottleIntegration(transaction_style=transaction_style)
        ]
    )
    events = capture_events()

    client = get_client()
    response = client.get(url)
    assert response[1] == "200 OK"

    (event,) = events
    # We use endswith() because in Python 2.7 it is "test_bottle.hi"
    # and in later Pythons "test_bottle.app.<locals>.hi"
    assert event["transaction"].endswith(expected_transaction)
    assert event["transaction_info"] == {"source": expected_source}


@pytest.mark.parametrize("debug", (True, False), ids=["debug", "nodebug"])
@pytest.mark.parametrize("catchall", (True, False), ids=["catchall", "nocatchall"])
def test_errors(
    debugg_ai_init, capture_exceptions, capture_events, app, debug, catchall, get_client
):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    app.catchall = catchall
    set_debug(mode=debug)

    exceptions = capture_exceptions()
    events = capture_events()

    @app.route("/")
    def index():
        1 / 0

    client = get_client()
    try:
        client.get("/")
    except ZeroDivisionError:
        pass

    (exc,) = exceptions
    assert isinstance(exc, ZeroDivisionError)

    (event,) = events
    assert event["exception"]["values"][0]["mechanism"]["type"] == "bottle"
    assert event["exception"]["values"][0]["mechanism"]["handled"] is False


def test_large_json_request(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    data = {"foo": {"bar": "a" * 2000}}

    @app.route("/", method="POST")
    def index():
        import bottle

        assert bottle.request.json == data
        assert bottle.request.body.read() == json.dumps(data).encode("ascii")
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()
    response = client.get("/")

    response = client.post("/", content_type="application/json", data=json.dumps(data))
    assert response[1] == "200 OK"

    (event,) = events
    assert event["_meta"]["request"]["data"]["foo"]["bar"] == {
        "": {"len": 2000, "rem": [["!limit", "x", 1021, 1024]]}
    }
    assert len(event["request"]["data"]["foo"]["bar"]) == 1024


@pytest.mark.parametrize("data", [{}, []], ids=["empty-dict", "empty-list"])
def test_empty_json_request(debugg_ai_init, capture_events, app, data, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    @app.route("/", method="POST")
    def index():
        import bottle

        assert bottle.request.json == data
        assert bottle.request.body.read() == json.dumps(data).encode("ascii")
        # assert not bottle.request.forms
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()
    response = client.post("/", content_type="application/json", data=json.dumps(data))
    assert response[1] == "200 OK"

    (event,) = events
    assert event["request"]["data"] == data


def test_medium_formdata_request(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    data = {"foo": "a" * 2000}

    @app.route("/", method="POST")
    def index():
        import bottle

        assert bottle.request.forms["foo"] == data["foo"]
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()
    response = client.post("/", data=data)
    assert response[1] == "200 OK"

    (event,) = events
    assert event["_meta"]["request"]["data"]["foo"] == {
        "": {"len": 2000, "rem": [["!limit", "x", 1021, 1024]]}
    }
    assert len(event["request"]["data"]["foo"]) == 1024


@pytest.mark.parametrize("input_char", ["a", b"a"])
def test_too_large_raw_request(
    debugg_ai_init, input_char, capture_events, app, get_client
):
    debugg_ai_init(
        integrations=[bottle_debugg-ai.BottleIntegration()], max_request_body_size="small"
    )

    data = input_char * 2000

    @app.route("/", method="POST")
    def index():
        import bottle

        if isinstance(data, bytes):
            assert bottle.request.body.read() == data
        else:
            assert bottle.request.body.read() == data.encode("ascii")
        assert not bottle.request.json
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()
    response = client.post("/", data=data)
    assert response[1] == "200 OK"

    (event,) = events
    assert event["_meta"]["request"]["data"] == {"": {"rem": [["!config", "x"]]}}
    assert not event["request"]["data"]


def test_files_and_form(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(
        integrations=[bottle_debugg-ai.BottleIntegration()], max_request_body_size="always"
    )

    data = {"foo": "a" * 2000, "file": (BytesIO(b"hello"), "hello.txt")}

    @app.route("/", method="POST")
    def index():
        import bottle

        assert list(bottle.request.forms) == ["foo"]
        assert list(bottle.request.files) == ["file"]
        assert not bottle.request.json
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()
    response = client.post("/", data=data)
    assert response[1] == "200 OK"

    (event,) = events
    assert event["_meta"]["request"]["data"]["foo"] == {
        "": {"len": 2000, "rem": [["!limit", "x", 1021, 1024]]}
    }
    assert len(event["request"]["data"]["foo"]) == 1024

    assert event["_meta"]["request"]["data"]["file"] == {
        "": {
            "rem": [["!raw", "x"]],
        }
    }
    assert not event["request"]["data"]["file"]


def test_json_not_truncated_if_max_request_body_size_is_always(
    debugg_ai_init, capture_events, app, get_client
):
    debugg_ai_init(
        integrations=[bottle_debugg-ai.BottleIntegration()], max_request_body_size="always"
    )

    data = {
        "key{}".format(i): "value{}".format(i) for i in range(MAX_DATABAG_BREADTH + 10)
    }

    @app.route("/", method="POST")
    def index():
        import bottle

        assert bottle.request.json == data
        assert bottle.request.body.read() == json.dumps(data).encode("ascii")
        capture_message("hi")
        return "ok"

    events = capture_events()

    client = get_client()

    response = client.post("/", content_type="application/json", data=json.dumps(data))
    assert response[1] == "200 OK"

    (event,) = events
    assert event["request"]["data"] == data


@pytest.mark.parametrize(
    "integrations",
    [
        [bottle_debugg-ai.BottleIntegration()],
        [bottle_debugg-ai.BottleIntegration(), LoggingIntegration(event_level="ERROR")],
    ],
)
def test_errors_not_reported_twice(
    debugg_ai_init, integrations, capture_events, app, get_client
):
    debugg_ai_init(integrations=integrations)

    app.catchall = False

    logger = logging.getLogger("bottle.app")

    @app.route("/")
    def index():
        try:
            1 / 0
        except Exception as e:
            logger.exception(e)
            raise e

    events = capture_events()

    client = get_client()
    with pytest.raises(ZeroDivisionError):
        client.get("/")

    assert len(events) == 1


def test_mount(app, capture_exceptions, capture_events, debugg_ai_init, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    app.catchall = False

    def crashing_app(environ, start_response):
        1 / 0

    app.mount("/wsgi/", crashing_app)

    client = Client(app)

    exceptions = capture_exceptions()
    events = capture_events()

    with pytest.raises(ZeroDivisionError) as exc:
        client.get("/wsgi/")

    (error,) = exceptions

    assert error is exc.value

    (event,) = events
    assert event["exception"]["values"][0]["mechanism"]["type"] == "bottle"
    assert event["exception"]["values"][0]["mechanism"]["handled"] is False


def test_error_in_errorhandler(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])

    set_debug(False)
    app.catchall = True

    @app.route("/")
    def index():
        raise ValueError()

    @app.error(500)
    def error_handler(err):
        1 / 0

    events = capture_events()

    client = get_client()

    with pytest.raises(ZeroDivisionError):
        client.get("/")

    event1, event2 = events

    (exception,) = event1["exception"]["values"]
    assert exception["type"] == "ValueError"

    exception = event2["exception"]["values"][0]
    assert exception["type"] == "ZeroDivisionError"


def test_bad_request_not_captured(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])
    events = capture_events()

    @app.route("/")
    def index():
        abort(400, "bad request in")

    client = get_client()

    client.get("/")

    assert not events


def test_no_exception_on_redirect(debugg_ai_init, capture_events, app, get_client):
    debugg_ai_init(integrations=[bottle_debugg-ai.BottleIntegration()])
    events = capture_events()

    @app.route("/")
    def index():
        redirect("/here")

    @app.route("/here")
    def here():
        return "here"

    client = get_client()

    client.get("/")

    assert not events


def test_span_origin(
    debugg_ai_init,
    get_client,
    capture_events,
):
    debugg_ai_init(
        integrations=[bottle_debugg-ai.BottleIntegration()],
        traces_sample_rate=1.0,
    )
    events = capture_events()

    client = get_client()
    client.get("/message")

    (_, event) = events

    assert event["contexts"]["trace"]["origin"] == "auto.http.bottle"


@pytest.mark.parametrize("raise_error", [True, False])
@pytest.mark.parametrize(
    ("integration_kwargs", "status_code", "should_capture"),
    (
        ({}, None, False),
        ({}, 400, False),
        ({}, 451, False),  # Highest 4xx status code
        ({}, 500, True),
        ({}, 511, True),  # Highest 5xx status code
        ({"failed_request_status_codes": set()}, 500, False),
        ({"failed_request_status_codes": set()}, 511, False),
        ({"failed_request_status_codes": {404, *range(500, 600)}}, 404, True),
        ({"failed_request_status_codes": {404, *range(500, 600)}}, 500, True),
        ({"failed_request_status_codes": {404, *range(500, 600)}}, 400, False),
    ),
)
def test_failed_request_status_codes(
    debugg_ai_init,
    capture_events,
    integration_kwargs,
    status_code,
    should_capture,
    raise_error,
):
    debugg_ai_init(integrations=[BottleIntegration(**integration_kwargs)])
    events = capture_events()

    app = Bottle()

    @app.route("/")
    def handle():
        if status_code is not None:
            response = HTTPResponse(status=status_code)
            if raise_error:
                raise response
            else:
                return response
        return "OK"

    client = Client(app, Response)
    response = client.get("/")

    expected_status = 200 if status_code is None else status_code
    assert response.status_code == expected_status

    if should_capture:
        (event,) = events
        assert event["exception"]["values"][0]["type"] == "HTTPResponse"
    else:
        assert not events


def test_failed_request_status_codes_non_http_exception(debugg_ai_init, capture_events):
    """
    If an exception, which is not an instance of HTTPResponse, is raised, it should be captured, even if
    failed_request_status_codes is empty.
    """
    debugg_ai_init(integrations=[BottleIntegration(failed_request_status_codes=set())])
    events = capture_events()

    app = Bottle()

    @app.route("/")
    def handle():
        1 / 0

    client = Client(app, Response)

    try:
        client.get("/")
    except ZeroDivisionError:
        pass

    (event,) = events
    assert event["exception"]["values"][0]["type"] == "ZeroDivisionError"
