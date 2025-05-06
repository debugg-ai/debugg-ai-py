import os
import debugg_ai_sdk
from debugg_ai_sdk.integrations.aws_lambda import AwsLambdaIntegration


debugg_ai_sdk.init(
    dsn=os.environ.get("DEBUGGAI_INGEST_URL"),
    traces_sample_rate=None,  # this is the default, just added for clarity
    integrations=[AwsLambdaIntegration()],
)


def handler(event, context):
    raise Exception("Oh!")
