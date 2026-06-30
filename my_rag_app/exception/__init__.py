"""

This module defines custom exception classes and error-handling utilities tailored
to the needs of a data science pipeline. It helps standardize error reporting, improve
debugging, and provide meaningful feedback during model training, data preprocessing,
and prediction processes.

Classes:
    DataValidationError: Raised when input data fails validation checks.
    ModelTrainingError: Raised during errors in the model training phase, such as convergence issues or invalid configurations.
    PredictionError: Raised when the prediction pipeline encounters issues, such as missing features or incompatible input formats.
    PipelineExecutionError: Raised for generic errors occurring during pipeline execution.

Usage:
    Import and use the exceptions in various stages of the data science pipeline:

    Example:
        ```python
        from exception import DataValidationError, ModelTrainingError

        try:
            validate_data(input_data)
        except DataValidationError as e:
            logger.error(f"Data validation failed: {e}")
            raise
        ```

Features:
    - Custom exceptions for specific pipeline stages, ensuring meaningful error reporting.
    - Enables targeted exception handling, reducing debugging time.
    - Provides a consistent structure for error messages across the project.

Purpose:
    - To define project-specific exceptions for common error scenarios in the pipeline.
    - To improve the robustness and reliability of the pipeline by enabling clear error handling.
    - To make the debugging process more intuitive by raising descriptive errors.

Examples:
    - **Data Validation**: Raise a `DataValidationError` if the input data schema is incorrect or missing required fields.
    - **Model Training**: Raise a `ModelTrainingError` if the model fails to converge due to invalid hyperparameters.
    - **Prediction**: Raise a `PredictionError` when incompatible input data is passed to the model.

Additional Notes:
    - Use these exceptions in conjunction with logging to provide detailed error information.
    - Ensure that custom exceptions are raised with meaningful messages to assist in debugging and error resolution.
"""

import logging
import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    # Extract traceback details (exception information)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message string with file name, line number, and the actual error
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {error!s}"

    # Log the error for better tracking
    logging.error(error_message)

    return error_message


class MyException(Exception):
    """
    Custom exception class for handling errors in the US visa application.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the USvisaException with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        # Call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message
