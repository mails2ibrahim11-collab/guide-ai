import logging
import sys
from datetime import datetime

# ================================================================
# GuideAI — Central Logger
# All files import from here. One consistent format across the app.
# ================================================================

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-20s  %(message)s"
DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger that writes to the terminal (stdout).
    Call this at the top of each module:
        from logger import get_logger
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Don't add duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# ================================================================
# Log level guide used throughout the project:
#
#   log.debug(...)    → internal step details (loop progress, counts)
#   log.info(...)     → normal checkpoints (startup, request received)
#   log.warning(...)  → recoverable issues (chunk skipped, fallback used)
#   log.error(...)    → failures that returned an error response
#   log.critical(...) → startup failures (DB init failed, PDF missing)
# ================================================================