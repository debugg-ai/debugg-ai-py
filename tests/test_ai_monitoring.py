import pytest

import debugg_ai_sdk
from debugg_ai_sdk.ai.monitoring import ai_track


def test_ai_track(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    @ai_track("my tool")
    def tool(**kwargs):
        pass

    @ai_track("some test pipeline")
    def pipeline():
        tool()

    with debugg_ai_sdk.start_transaction():
        pipeline()

    transaction = events[0]
    assert transaction["type"] == "transaction"
    assert len(transaction["spans"]) == 2
    spans = transaction["spans"]

    ai_pipeline_span = spans[0] if spans[0]["op"] == "ai.pipeline" else spans[1]
    ai_run_span = spans[0] if spans[0]["op"] == "ai.run" else spans[1]

    assert ai_pipeline_span["description"] == "some test pipeline"
    assert ai_run_span["description"] == "my tool"


def test_ai_track_with_tags(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    @ai_track("my tool")
    def tool(**kwargs):
        pass

    @ai_track("some test pipeline")
    def pipeline():
        tool()

    with debugg_ai_sdk.start_transaction():
        pipeline(debugg_ai_tags={"user": "colin"}, debugg_ai_data={"some_data": "value"})

    transaction = events[0]
    assert transaction["type"] == "transaction"
    assert len(transaction["spans"]) == 2
    spans = transaction["spans"]

    ai_pipeline_span = spans[0] if spans[0]["op"] == "ai.pipeline" else spans[1]
    ai_run_span = spans[0] if spans[0]["op"] == "ai.run" else spans[1]

    assert ai_pipeline_span["description"] == "some test pipeline"
    print(ai_pipeline_span)
    assert ai_pipeline_span["tags"]["user"] == "colin"
    assert ai_pipeline_span["data"]["some_data"] == "value"
    assert ai_run_span["description"] == "my tool"


@pytest.mark.asyncio
async def test_ai_track_async(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    @ai_track("my async tool")
    async def async_tool(**kwargs):
        pass

    @ai_track("some async test pipeline")
    async def async_pipeline():
        await async_tool()

    with debugg_ai_sdk.start_transaction():
        await async_pipeline()

    transaction = events[0]
    assert transaction["type"] == "transaction"
    assert len(transaction["spans"]) == 2
    spans = transaction["spans"]

    ai_pipeline_span = spans[0] if spans[0]["op"] == "ai.pipeline" else spans[1]
    ai_run_span = spans[0] if spans[0]["op"] == "ai.run" else spans[1]

    assert ai_pipeline_span["description"] == "some async test pipeline"
    assert ai_run_span["description"] == "my async tool"


@pytest.mark.asyncio
async def test_ai_track_async_with_tags(debugg_ai_init, capture_events):
    debugg_ai_init(traces_sample_rate=1.0)
    events = capture_events()

    @ai_track("my async tool")
    async def async_tool(**kwargs):
        pass

    @ai_track("some async test pipeline")
    async def async_pipeline():
        await async_tool()

    with debugg_ai_sdk.start_transaction():
        await async_pipeline(
            debugg_ai_tags={"user": "czyber"}, debugg_ai_data={"some_data": "value"}
        )

    transaction = events[0]
    assert transaction["type"] == "transaction"
    assert len(transaction["spans"]) == 2
    spans = transaction["spans"]

    ai_pipeline_span = spans[0] if spans[0]["op"] == "ai.pipeline" else spans[1]
    ai_run_span = spans[0] if spans[0]["op"] == "ai.run" else spans[1]

    assert ai_pipeline_span["description"] == "some async test pipeline"
    assert ai_pipeline_span["tags"]["user"] == "czyber"
    assert ai_pipeline_span["data"]["some_data"] == "value"
    assert ai_run_span["description"] == "my async tool"
