from my_rag_app.constants import PROMPT_VERSION, SYSTEM_PROMPT_V3
from my_rag_app.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Assembles the messages list sent to the LLM from context, query, and chat history."""

    def get_system_prompt(self) -> str:
        """Return the system prompt used for every request."""
        return SYSTEM_PROMPT_V3

    def get_user_prompt(self, query: str, context: str) -> str:
        """Build the user message combining the query and retrieved context."""
        if not context.strip():
            return f"Question: {query}\n\nNo relevant emails were found in the knowledge base for this question."
        return f"Context (relevant emails):\n\n{context}\n\nQuestion: {query}"

    def build(
        self,
        query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the full messages list to send to the LLM.

        Prepends the last 6 chat history turns (3 user + 3 assistant) before
        the current query so the LLM has conversational context for follow-ups
        like 'show me the mail' or 'who sent that?'.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": self.get_system_prompt()}]

        # Include last 6 turns (3 exchanges) — enough for follow-up resolution
        # without bloating the context window on long conversations
        if chat_history:
            messages.extend(chat_history[-6:])

        messages.append({"role": "user", "content": self.get_user_prompt(query, context)})

        logger.info(
            "Prompt built | version=%s query_len=%d context_len=%d history_turns=%d",
            PROMPT_VERSION,
            len(query),
            len(context),
            len(chat_history) if chat_history else 0,
        )
        return messages
