import sys
import logging
import warnings

from debugg_ai_sdk import get_client
from debugg_ai_sdk.client import _client_init_debug
from debugg_ai_sdk.utils import logger
from logging import LogRecord


class _DebugFilter(logging.Filter):
    def filter(self, record):
        # type: (LogRecord) -> bool
        if _client_init_debug.get(False):
            return True

        return get_client().options["debug"]


def init_debug_support():
    # type: () -> None
    if not logger.handlers:
        configure_logger()


def configure_logger():
    # type: () -> None
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(" [debugg-ai] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_DebugFilter())


def configure_debug_hub():
    # type: () -> None
    warnings.warn(
        "configure_debug_hub is deprecated. Please remove calls to it, as it is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )
