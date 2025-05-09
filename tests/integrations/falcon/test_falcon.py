import logging

import pytest

import falcon
import falcon.testing
import debugg_ai_sdk
from debugg_ai_sdk.integrations.falcon import FalconIntegration
from debugg_ai_sdk.integrations.logging import LoggingIntegration
from debugg_ai_sdk.utils import parse_version


try:
    import falcon.asgi
except ImportError:
    pass
else:
    import falcon.inspect  # We only need this module for the ASGI test


FALCON_VERSION = parse_version(falcon.__version__)


@pytest.fixture
def make_app(debugg_ai_init):
    def inner():
        class MessageResource:
            def on_get(self, req, resp):
                debugg_ai_sdk.capture_message("hi")
                resp.media = "hi"

        class MessageByIdResource:
            def on_get(self, req, resp, message_id):
                debugg_ai_sdk.capture_message("hi")
                resp.media = "hi"

        class CustomError(Exception):
            pass

        class CustomErrorResource:
            def on_get(self, req, resp):
                raise CustomError()

        def custom_error_handler(*args, **kwargs):
            raise falcon.HTTPError(status=falcon.HTTP_400)

        app = falcon.API()
        app.add_route("/message", MessageResource())
        app.add_route("/message/{message_id:int}", MessageByIdResource())
        app.add_route("/custom-error", CustomErrorResource())

        app.add_error_handler(CustomError, custom_error_handler)

        return app

    return inner


@pytest.fixture
def make_client(make_app):
    def inner():
        app = make_app()
        return falcon.testing.TestClient(app)

    return inner


def test_has_context(debugg_ai_init, capture_events, make_client):
    debugg_ai_init(integrations=[FalconIntegration()])
    events = capture_events()

    client = make_client()
    response = client.simulate_get("/message")
    assert response.status == falcon.HTTP_200

    (event,) = events
    assert event["transaction"] == "/message"  # Falcon URI template
    assert "data" not in event["request"]
    assert event["request"]["url"] == "http://falconframework.org/message"


@pytest.mark.parametrize(
    "url,transaction_style,expected_transaction,expected_source",
    [
        ("/message", "uri_template", "/message", "route"),
        ("/message", "path", "/message", "url"),
        ("/message/123456", "uri_template", "/message/{message_id:int}", "route"),
        ("/message/123456", "path", "/message/123456", "url"),
    ],
)
def test_transaction_style(
    debugg_ai_init,
    make_client,
    capture_events,
    url,
    transaction_style,
    expected_transaction,
    expected_source,
):
    integration = FalconIntegration(transaction_style=transaction_style)
    debugg_ai_init(integrations=[integration])
    events = capture_events()

    client = make_client()
    response = client.simulate_get(url)
    assert response.status == falcon.HTTP_200

    (event,) = events
    assert event["transaction"] == expected_transaction
    assert event["transaction_info"] == {"source": expected_source}


def test_unhandled_errors(debugg_ai_init, capture_exceptions, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_get(self, req, resp):
            1 / 0

    app = falcon.API()
    app.add_route("/", Resource())

    exceptions = capture_exceptions()
    events = capture_events()

    client = falcon.testing.TestClient(app)

    try:
        client.simulate_get("/")
    except ZeroDivisionError:
        pass

    (exc,) = exceptions
    assert isinstance(exc, ZeroDivisionError)

    (event,) = events
    assert event["exception"]["values"][0]["mechanism"]["type"] == "falcon"
    assert " by zero" in event["exception"]["values"][0]["value"]


def test_raised_5xx_errors(debugg_ai_init, capture_exceptions, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_get(self, req, resp):
            raise falcon.HTTPError(falcon.HTTP_502)

    app = falcon.API()
    app.add_route("/", Resource())

    exceptions = capture_exceptions()
    events = capture_events()

    client = falcon.testing.TestClient(app)
    client.simulate_get("/")

    (exc,) = exceptions
    assert isinstance(exc, falcon.HTTPError)

    (event,) = events
    assert event["exception"]["values"][0]["mechanism"]["type"] == "falcon"
    assert event["exception"]["values"][0]["type"] == "HTTPError"


def test_raised_4xx_errors(debugg_ai_init, capture_exceptions, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_get(self, req, resp):
            raise falcon.HTTPError(falcon.HTTP_400)

    app = falcon.API()
    app.add_route("/", Resource())

    exceptions = capture_exceptions()
    events = capture_events()

    client = falcon.testing.TestClient(app)
    client.simulate_get("/")

    assert len(exceptions) == 0
    assert len(events) == 0


def test_http_status(debugg_ai_init, capture_exceptions, capture_events):
    """
    This just demonstrates, that if Falcon raises a HTTPStatus with code 500
    (instead of a HTTPError with code 500) DebuggAI will not capture it.
    """
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_get(self, req, resp):
            raise falcon.http_status.HTTPStatus(falcon.HTTP_508)

    app = falcon.API()
    app.add_route("/", Resource())

    exceptions = capture_exceptions()
    events = capture_events()

    client = falcon.testing.TestClient(app)
    client.simulate_get("/")

    assert len(exceptions) == 0
    assert len(events) == 0


def test_falcon_large_json_request(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    data = {"foo": {"bar": "a" * 2000}}

    class Resource:
        def on_post(self, req, resp):
            assert req.media == data
            debugg_ai_sdk.capture_message("hi")
            resp.media = "ok"

    app = falcon.API()
    app.add_route("/", Resource())

    events = capture_events()

    client = falcon.testing.TestClient(app)
    response = client.simulate_post("/", json=data)
    assert response.status == falcon.HTTP_200

    (event,) = events
    assert event["_meta"]["request"]["data"]["foo"]["bar"] == {
        "": {"len": 2000, "rem": [["!limit", "x", 1021, 1024]]}
    }
    assert len(event["request"]["data"]["foo"]["bar"]) == 1024


@pytest.mark.parametrize("data", [{}, []], ids=["empty-dict", "empty-list"])
def test_falcon_empty_json_request(debugg_ai_init, capture_events, data):
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_post(self, req, resp):
            assert req.media == data
            debugg_ai_sdk.capture_message("hi")
            resp.media = "ok"

    app = falcon.API()
    app.add_route("/", Resource())

    events = capture_events()

    client = falcon.testing.TestClient(app)
    response = client.simulate_post("/", json=data)
    assert response.status == falcon.HTTP_200

    (event,) = events
    assert event["request"]["data"] == data


def test_falcon_raw_data_request(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    class Resource:
        def on_post(self, req, resp):
            debugg_ai_sdk.capture_message("hi")
            resp.media = "ok"

    app = falcon.API()
    app.add_route("/", Resource())

    events = capture_events()

    client = falcon.testing.TestClient(app)
    response = client.simulate_post("/", body="hi")
    assert response.status == falcon.HTTP_200

    (event,) = events
    assert event["request"]["headers"]["Content-Length"] == "2"
    assert event["request"]["data"] == ""


def test_logging(debugg_ai_init, capture_events):
    debugg_ai_init(
        integrations=[FalconIntegration(), LoggingIntegration(event_level="ERROR")]
    )

    logger = logging.getLogger()

    app = falcon.API()

    class Resource:
        def on_get(self, req, resp):
            logger.error("hi")
            resp.media = "ok"

    app.add_route("/", Resource())

    events = capture_events()

    client = falcon.testing.TestClient(app)
    client.simulate_get("/")

    (event,) = events
    assert event["level"] == "error"


def test_500(debugg_ai_init):
    debugg_ai_init(integrations=[FalconIntegration()])

    app = falcon.API()

    class Resource:
        def on_get(self, req, resp):
            1 / 0

    app.add_route("/", Resource())

    def http500_handler(ex, req, resp, params):
        debugg_ai_sdk.capture_exception(ex)
        resp.media = {"message": "DebuggAI error."}

    app.add_error_handler(Exception, http500_handler)

    client = falcon.testing.TestClient(app)
    response = client.simulate_get("/")

    assert response.json == {"message": "DebuggAI error."}


def test_error_in_errorhandler(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])

    app = falcon.API()

    class Resource:
        def on_get(self, req, resp):
            raise ValueError()

    app.add_route("/", Resource())

    def http500_handler(ex, req, resp, params):
        1 / 0

    app.add_error_handler(Exception, http500_handler)

    events = capture_events()

    client = falcon.testing.TestClient(app)

    with pytest.raises(ZeroDivisionError):
        client.simulate_get("/")

    (event,) = events

    last_ex_values = event["exception"]["values"][-1]
    assert last_ex_values["type"] == "ZeroDivisionError"
    assert last_ex_values["stacktrace"]["frames"][-1]["vars"]["ex"] == "ValueError()"


def test_bad_request_not_captured(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])
    events = capture_events()

    app = falcon.API()

    class Resource:
        def on_get(self, req, resp):
            raise falcon.HTTPBadRequest()

    app.add_route("/", Resource())

    client = falcon.testing.TestClient(app)

    client.simulate_get("/")

    assert not events


def test_does_not_leak_scope(debugg_ai_init, capture_events):
    debugg_ai_init(integrations=[FalconIntegration()])
    events = capture_events()

    debugg_ai_sdk.get_isolation_scope().set_tag("request_data", False)

    app = falcon.API()

    class Resource:
        def on_get(self, req, resp):
            debugg_ai_sdk.get_isolation_scope().set_tag("request_data", True)

            def generator():
                for row in range(1000):
                    assert debugg_ai_sdk.get_isolation_scope()._tags["request_data"]

                    yield (str(row) + "\n").encode()

            resp.stream = generator()

    app.add_route("/", Resource())

    client = falcon.testing.TestClient(app)
    response = client.simulate_get("/")

    expected_response = "".join(str(row) + "\n" for row in range(1000))
    assert response.text == expected_response
    assert not events
    assert not debugg_ai_sdk.get_isolation_scope()._tags["request_data"]


@pytest.mark.skipif(
    not hasattr(falcon, "asgi"), reason="This Falcon version lacks ASGI support."
)
def test_falcon_not_breaking_asgi(debugg_ai_init):
    """
    This test simply verifies that the Falcon integration does not break ASGI
    Falcon apps.

    The test does not verify ASGI Falcon support, since our Falcon integration
    currently lacks support for ASGI Falcon apps.
    """
    debugg_ai_init(integrations=[FalconIntegration()])

    asgi_app = falcon.asgi.App()

    try:
        falcon.inspect.inspect_app(asgi_app)
    except TypeError:
        pytest.fail("Falcon integration causing errors in ASGI apps.")


@pytest.mark.skipif(
    (FALCON_VERSION or ()) < (3,),
    reason="The DebuggAI Falcon integration only supports custom error handlers on Falcon 3+",
)
def test_falcon_custom_error_handler(debugg_ai_init, make_app, capture_events):
    """
    When a custom error handler handles what otherwise would have resulted in a 5xx error,
    changing the HTTP status to a non-5xx status, no error event should be sent to DebuggAI.
    """
    debugg_ai_init(integrations=[FalconIntegration()])
    events = capture_events()

    app = make_app()
    client = falcon.testing.TestClient(app)

    client.simulate_get("/custom-error")

    assert len(events) == 0


def test_span_origin(debugg_ai_init, capture_events, make_client):
    debugg_ai_init(
        integrations=[FalconIntegration()],
        traces_sample_rate=1.0,
    )
    events = capture_events()

    client = make_client()
    client.simulate_get("/message")

    (_, event) = events

    assert event["contexts"]["trace"]["origin"] == "auto.http.falcon"


def test_falcon_request_media(debugg_ai_init):
    # test_passed stores whether the test has passed.
    test_passed = False

    # test_failure_reason stores the reason why the test failed
    # if test_passed is False. The value is meaningless when
    # test_passed is True.
    test_failure_reason = "test endpoint did not get called"

    class DebuggAICaptureMiddleware:
        def process_request(self, _req, _resp):
            # This capture message forces Falcon event processors to run
            # before the request handler runs
            debugg_ai_sdk.capture_message("Processing request")

    class RequestMediaResource:
        def on_post(self, req, _):
            nonlocal test_passed, test_failure_reason
            raw_data = req.bounded_stream.read()

            # If the raw_data is empty, the request body stream
            # has been exhausted by the SDK. Test should fail in
            # this case.
            test_passed = raw_data != b""
            test_failure_reason = "request body has been read"

    debugg_ai_init(integrations=[FalconIntegration()])

    try:
        app_class = falcon.App  # Falcon ≥3.0
    except AttributeError:
        app_class = falcon.API  # Falcon <3.0

    app = app_class(middleware=[DebuggAICaptureMiddleware()])
    app.add_route("/read_body", RequestMediaResource())

    client = falcon.testing.TestClient(app)

    client.simulate_post("/read_body", json={"foo": "bar"})

    # Check that simulate_post actually calls the resource, and
    # that the SDK does not exhaust the request body stream.
    assert test_passed, test_failure_reason
