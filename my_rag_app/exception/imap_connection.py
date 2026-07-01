class ImapConnectionError(ConnectionError):
    """Raised when a connection to the IMAP server cannot be established."""

    def __init__(self, host: str, last_error: Exception | None = None):
        message = f"Could not connect to IMAP host {host}"
        super().__init__(message)
        self.host = host
        self.last_error = last_error


class ImapSearchError(RuntimeError):
    """Raised when an IMAP SEARCH operation fails."""

    def __init__(self, status: str):
        message = f"IMAP SEARCH failed with status {status}"
        super().__init__(message)
        self.status = status


class ImapConfigError(OSError):
    """Raised when required IMAP configuration is missing."""

    def __init__(self) -> None:
        super().__init__("Missing required IMAP credentials (IMAP_HOST, EMAIL_ADDR, EMAIL_PASSWORD)")
