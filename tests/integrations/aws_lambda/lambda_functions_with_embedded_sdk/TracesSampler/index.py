import json
import os
import debugg_ai_sdk
from debugg_ai_sdk.integrations.aws_lambda import AwsLambdaIntegration

# Global variables to store sampling context for verification
sampling_context_data = {
    "aws_event_present": False,
    "aws_context_present": False,
    "event_data": None,
}


def trace_sampler(sampling_context):
    # Store the sampling context for verification
    global sampling_context_data

    # Check if aws_event and aws_context are in the sampling_context
    if "aws_event" in sampling_context:
        sampling_context_data["aws_event_present"] = True
        sampling_context_data["event_data"] = sampling_context["aws_event"]

    if "aws_context" in sampling_context:
        sampling_context_data["aws_context_present"] = True

    print("Sampling context data:", sampling_context_data)
    return 1.0  # Always sample


debugg_ai_sdk.init(
    dsn=os.environ.get("DEBUGGAI_INGEST_URL"),
    traces_sample_rate=1.0,
    traces_sampler=trace_sampler,
    integrations=[AwsLambdaIntegration()],
)


def handler(event, context):
    # Return the sampling context data for verification
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Hello from Lambda with embedded Sentry SDK!",
                "event": event,
                "sampling_context_data": sampling_context_data,
            }
        ),
    }
