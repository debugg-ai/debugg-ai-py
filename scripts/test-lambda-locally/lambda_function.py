import logging
import os
import debugg_ai_sdk

from debugg_ai_sdk.integrations.aws_lambda import AwsLambdaIntegration
from debugg_ai_sdk.integrations.logging import LoggingIntegration

def lambda_handler(event, context):
    debugg_ai_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        attach_stacktrace=True,
        integrations=[
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            AwsLambdaIntegration(timeout_warning=True)
        ],
        traces_sample_rate=1.0,
        debug=True,
    )

    try:
        my_dict = {"a" : "test"}
        value = my_dict["b"] # This should raise exception
    except:
        logging.exception("Key Does not Exists")
        raise
