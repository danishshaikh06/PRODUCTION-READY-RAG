"""

This module provides centralized logging utilities for the data science pipeline.
It standardizes logging practices, ensures consistency across components, and facilitates
easy debugging and monitoring of the pipeline's execution, including data preprocessing,
model training, evaluation, and predictions.

Functions:
    setup_logging: Configures the logging system, including log format, level, and output destinations.
    get_logger: Returns a logger instance for a specific module or stage of the pipeline.

Features:
    - Centralized logging configuration to maintain consistency.
    - Support for different log levels (INFO, DEBUG, WARNING, ERROR, CRITICAL).
    - Ability to write logs to files, console, or external monitoring systems.
    - Timestamped log entries for accurate tracking of events.
    - Integration with custom exception handling for detailed error reporting.

Usage:
    Use this module to log messages in a standardized manner across the project:

    Example:
        ```python
        from src.logging import logger

        logger.info("Starting the model training process...")
        logger.error("An error occurred during data validation.")
        ```

Purpose:
    - To provide a standardized mechanism for logging messages throughout the data science pipeline.
    - To assist in debugging by capturing detailed logs of each pipeline stage.
    - To enable seamless integration with monitoring and alerting systems.

Best Practices:
    - Use appropriate log levels to categorize messages (e.g., DEBUG for detailed information, ERROR for issues).
    - Ensure logs include sufficient context, such as function names or input details, to aid debugging.
    - Regularly monitor log files for anomalies or errors in the pipeline.

Additional Notes:
    - The `setup_logging` function can be configured to write logs to multiple destinations, such as files or cloud services.
    - The module can be extended to integrate with third-party monitoring tools like Elasticsearch, Splunk, or Datadog.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")


def configure_logger() -> None:
    """
    Configure root logger with file and console handlers.
    """

    root_logger = logging.getLogger()

    # Prevent duplicate handlers when module is imported multiple times
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Configure logging when module is imported
configure_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance.

    Example:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
