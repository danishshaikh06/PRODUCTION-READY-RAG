from my_rag_app.logger import get_logger
from my_rag_app.constants import SYSTEM_PROMPT_V1, PROMPT_VERSION


logger = get_logger(__name__)


SYSTEM_PROMPT_V1 = SYSTEM_PROMPT_V1
PROMPT_VERSION = PROMPT_VERSION

class PromptBuilder:
    """
    Builds the final messages list sent to the LLM, combining the system
    prompt with the user's query and assembled context.
    """

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_V1

    def get_user_prompt(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                f"Question: {query}\n\n"
                f"No relevant emails were found in the knowledge base for this question."
            )
        return f"Context (relevant emails):\n\n{context}\n\nQuestion: {query}"

    def build(self, query: str, context: str) -> list[dict]:
        if not query or not query.strip():
            logger.warning("build called with empty query")

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_user_prompt(query, context)},
        ]
        logger.info(
            "Prompt built | version=%s query_len=%d context_len=%d",
            PROMPT_VERSION, len(query or ""), len(context or ""),
        )
        return messages