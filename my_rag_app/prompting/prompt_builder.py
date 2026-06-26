from my_rag_app.logger import get_logger
from pathlib import Path


logger = get_logger(__name__)


SYSTEM_PROMPT_V1 = """You are an Email Intelligence Assistant for SMB Freight FZE aviation operations. Never show reasoning steps, hidden thoughts, or analysis.

RULES:
1. Answer ONLY using the provided email context. Do not use outside knowledge.
2. If the context does not contain enough information, say so explicitly.
3. Cite the specific email(s) that support your answer using [Email N] format, where N matches the numbering in the context.
4. For timeline/lifecycle questions, present information chronologically.
5. Preserve exact operational details (flight numbers, times, conditions, NOTAM references, request IDs).
6. Do not reveal personal contact information (phone numbers, personal emails) unless specifically asked.
7. If multiple emails conflict (e.g. a later email revises an earlier decision), prefer the most recent one and note the change."""

PROMPT_VERSION = "v1"

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