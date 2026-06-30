class ImapConnectionError(ConnectionError):
    def __init__(self, host: str, last_error: Exception | None = None):
        message = f"Could not connect to IMAP host {host}"
        super().__init__(message)
        self.host = host
        self.last_error = last_error

class ImapSearchError(RuntimeError):
    def __init__(self, status: str):
        message = f"IMAP SEARCH failed with status {status}"
        super().__init__(message)
        self.status = status

class ImapConfigError(OSError):
    def __init__(self):
        super().__init__(
            "Missing required IMAP credentials (IMAP_HOST, EMAIL_ADDR, EMAIL_PASSWORD)"
        )