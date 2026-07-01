import time

import requests
import tiktoken

from my_rag_app.constants import LLM_BASE_URL, LLM_MODEL, LLM_REQUEST_TIMEOUT_SECONDS, TOKENIZER_ENCODING
from my_rag_app.entity.reports import LLMResponse
from my_rag_app.exception.model import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRequestError,
    LLMResponseParseError,
    LLMTimeoutError,
    PromptError,
)
from my_rag_app.logger import get_logger

logger = get_logger(__name__)


# LLMClient — talks to a local Ollama server via its native /api/chat endpoint
class LLMClient:
    """Client for generating responses from a local Ollama model."""

    def __init__(self, model_name: str = LLM_MODEL, base_url: str = LLM_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        # Native /api/chat endpoint, NOT /v1/chat/completions — the OpenAI-compat
        # layer unreliably ignores "think": false for Qwen3.5-family models,
        # causing the entire token budget to be burned on hidden reasoning
        # tokens with an empty visible response. The native endpoint correctly
        # honors think as a top-level parameter. See: ollama/ollama#14793
        self._endpoint = f"{self.base_url}/api/chat"

        try:
            self._encoder = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception as e:
            logger.warning(
                "Could not load tiktoken encoding — falling back to char-count approximation | error=%s",
                e,
            )
            self._encoder = None

    # Token counting (approximation — no real endpoint for this in Ollama)
    def count_tokens(self, text: str) -> int:
        """Estimate the token count of a text string."""
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(text) // 4

    # Generation
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response from the LLM given a list of chat messages."""
        if not messages:
            logger.error("generate called with empty messages list")
            raise PromptError()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        input_text = "\n".join(m.get("content", "") for m in messages)
        input_tokens = self.count_tokens(input_text)

        start = time.monotonic()
        try:
            response = requests.post(self._endpoint, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.ConnectionError as e:
            logger.exception(
                "Could not connect to Ollama at %s — is it running? | error=%s",
                self.base_url,
            )
            raise LLMConnectionError(self.base_url) from e
        except requests.exceptions.Timeout as e:
            logger.exception(
                "Ollama request timed out after %ds | error=%s",
                LLM_REQUEST_TIMEOUT_SECONDS,
            )
            raise LLMTimeoutError(LLM_REQUEST_TIMEOUT_SECONDS) from e

        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code == 404:
            logger.error(
                "Model '%s' not found on Ollama server. Run: ollama pull %s",
                self.model_name,
                self.model_name,
            )
            raise LLMModelNotFoundError(self.model_name)

        if response.status_code != 200:
            logger.error(
                "Ollama returned status %d | body=%s",
                response.status_code,
                response.text[:500],
            )
            raise LLMRequestError(response.status_code, response.text[:500])
        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, ValueError) as e:
            logger.exception(
                "Unexpected response shape from Ollama | error=%s body=%s",
                response.text[:500],
            )
            raise LLMResponseParseError() from e

        # Native /api/chat reports real counts — prefer these over the tiktoken
        # approximation since they're exact, not estimated.
        input_tokens = data.get("prompt_eval_count", input_tokens)
        output_tokens = data.get("eval_count", self.count_tokens(content))

        logger.info(
            "Generation complete | model=%s input_tokens=%d output_tokens=%d latency_ms=%.0f",
            self.model_name,
            input_tokens,
            output_tokens,
            latency_ms,
        )

        return LLMResponse(
            content=content,
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
