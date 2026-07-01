from my_rag_app.constants import PROMPT_VERSION, SYSTEM_PROMPT_V1
from my_rag_app.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds the final messages list sent to the LLM, combining the system
    prompt with the user's query and assembled context.
    """

    def get_system_prompt(self) -> str:
        """Return the system prompt used for every request."""
        return SYSTEM_PROMPT_V1

    def get_user_prompt(self, query: str, context: str) -> str:
        """Build the user message combining the query and retrieved context."""
        if not context.strip():
            return f"Question: {query}\n\nNo relevant emails were found in the knowledge base for this question."
        return f"Context (relevant emails):\n\n{context}\n\nQuestion: {query}"

    def build(self, query: str, context: str) -> list[dict]:
        """Build the full messages list to send to the LLM."""
        if not query or not query.strip():
            logger.warning("build called with empty query")

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_user_prompt(query, context)},
        ]
        logger.info(
            "Prompt built | version=%s query_len=%d context_len=%d",
            PROMPT_VERSION,
            len(query or ""),
            len(context or ""),
        )
        return messages
