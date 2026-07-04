"""LLM client using a locally loaded HuggingFace model for GPU-accelerated inference."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from my_rag_app.constants import LLM_MODEL_V2, MAX_TOKEN_GENERATION,MODEL_REVISION
from my_rag_app.entity.reports import LLMResponse
from my_rag_app.exception.model import LLMResponseParseError
from my_rag_app.logger import get_logger

logger = get_logger(__name__)

class QwenClient:
    """GPU-accelerated LLM client using a locally loaded Qwen2.5-Instruct model.

    Loads the model once at instantiation and reuses it across all generate() calls.
    Drop-in replacement for the Ollama-based LLMClient — identical public interface.
    """

    def __init__(
        self,
        model_id: str = LLM_MODEL_V2,
    ) -> None:
        """Load the tokenizer and model onto the available device (GPU if present)."""
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading tokenizer | model=%s", model_id)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
            
        )

        logger.info("Loading model | model=%s device=%s", model_id, self.device)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.eval() # # It helps in removing the dropout layers and other training-specific layers from the model, which can improve inference speed and reduce memory usage.
        logger.info("Model ready | model=%s device=%s", model_id, self.device)

    def count_tokens(self, text: str) -> int:
        """Estimate token count using the model's own tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = MAX_TOKEN_GENERATION,
    ) -> LLMResponse:
        """Generate a response from the model given a list of chat messages.

        Converts messages to a prompt string via the tokenizer's chat template,
        runs inference, and returns an LLMResponse with content and usage stats.
        """
        if not messages:
            raise LLMResponseParseError()

        # Apply Qwen2.5-Instruct chat template to convert messages → prompt string
        prompt: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, #Append the assistant start token
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = inputs["input_ids"].shape[-1] # Count of tokens in the input prompt

        start = time.monotonic()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.monotonic() - start) * 1000

        # Decode only the newly generated tokens (not the prompt)
        new_token_ids = output_ids[0][input_tokens:]
        content = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        output_tokens = len(new_token_ids)

        if not content:
            raise LLMResponseParseError()

        logger.info(
            "Generation complete | model=%s input_tokens=%d output_tokens=%d latency_ms=%.0f device=%s",
            self.model_id,
            input_tokens,
            output_tokens,
            latency_ms,
            self.device,
        )

        return LLMResponse(
            content=content,
            model=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
load = QwenClient()