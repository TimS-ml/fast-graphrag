"""Custom exception classes for the fast-graphrag library.

This module defines specific exception types for handling errors in:
- Storage operations and data persistence
- Storage usage patterns and lifecycle management
- LLM (Large Language Model) service interactions
"""


class InvalidStorageError(Exception):
    """Exception raised when a storage operation is invalid or fails.

    This exception is typically raised when:
    - An operation is attempted on a storage backend in the wrong mode
    - A storage operation encounters an unexpected error
    - Data corruption or invalid state is detected in storage

    Attributes:
        message (str): Human-readable description of the error
    """

    def __init__(self, message: str = "Invalid storage operation"):
        """Initialize the InvalidStorageError with a custom message.

        Args:
            message (str): Description of what went wrong with the storage operation.
                Defaults to "Invalid storage operation".
        """
        self.message = message
        super().__init__(self.message)


class InvalidStorageUsageError(Exception):
    """Exception raised when storage is used incorrectly.

    This exception indicates a programming error in how storage is being used, such as:
    - Attempting to query storage that hasn't been initialized
    - Using storage operations in the wrong order or context
    - Violating storage lifecycle requirements (e.g., not calling upsert before query)

    Attributes:
        message (str): Human-readable description of the usage error
    """

    def __init__(self, message: str = "Invalid usage of the storage"):
        """Initialize the InvalidStorageUsageError with a custom message.

        Args:
            message (str): Description of the incorrect storage usage pattern.
                Defaults to "Invalid usage of the storage".
        """
        self.message = message
        super().__init__(self.message)


class LLMServiceNoResponseError(Exception):
    """Exception raised when an LLM service fails to return a response.

    This exception is raised when:
    - The LLM API call times out or returns empty response
    - The LLM service returns an error instead of a valid completion
    - Response parsing fails due to malformed or missing data

    Attributes:
        message (str): Human-readable description of the LLM response failure
    """

    def __init__(self, message: str = "LLM service did not provide a response"):
        """Initialize the LLMServiceNoResponseError with a custom message.

        Args:
            message (str): Description of why the LLM service failed to respond.
                Defaults to "LLM service did not provide a response".
        """
        self.message = message
        super().__init__(self.message)
