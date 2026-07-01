class MissingDBCredentialsError(OSError):
    """Throws an exception when database credential are missing"""

    def __init__(self, missing: list[str]):
        message = f"Missing required DB credentials in .env: {', '.join(missing)}"
        super().__init__(message)
        self.missing = missing
