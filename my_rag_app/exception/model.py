class InvalidPromptError(Exception):
    pass

class LLMConnectionError(Exception):
    def __init__(self, base_url: str):   
        message = f"Ollama unreachable at {base_url}"
        super().__init__(message)
        self.base_url = base_url

class LLMTimeoutError(Exception):
    def __init__(self, timeout_seconds: int):
        message = f"Ollama did not respond within {timeout_seconds}s"
        super().__init__(message)
        self.timeout_seconds = timeout_seconds

class LLMModelNotFoundError(Exception):
    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found"
        super().__init__(message)
        self.model_name = model_name

class LLMRequestError(Exception):
    def __init__(self, status_code: int, body: str | None = None):
        message = f"Ollama request failed with status {status_code}"
        super().__init__(message)
        self.status_code = status_code
        self.body = body

class LLMResponseParseError(Exception):
    def __init__(self, message: str = "Could not parse Ollama response"):
        super().__init__(message)