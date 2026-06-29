from my_rag_app.logger import get_logger
from pathlib import Path
import tiktoken
from my_rag_app.constants import CONTEXT_MAX_TOKENS, TOKENIZER_ENCODING

logger = get_logger(__name__)



# ContextBuilder
class ContextBuilder:
    """
    Assembles reranked search results + their full thread context into a
    single budget-limited string for the LLM prompt.

    Strategy:
      - Each reranked result's thread is expanded (via retriever.expand_threads)
      - Threads are deduplicated — a thread is only included once even if
        multiple reranked results belong to it
      - Threads are ordered by relevance (best matching result first);
        emails within a thread are ordered chronologically
      - Whole threads are dropped from the end if they would exceed the
        token budget — no mid-email truncation
    """

    def __init__(self, max_tokens: int = CONTEXT_MAX_TOKENS):
        self.max_tokens = max_tokens
        try:
            self._encoder = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception as e:
            logger.warning(
                "Could not load tiktoken encoding (%s) — falling back to "
                "char-count approximation | error=%s", TOKENIZER_ENCODING, e,
            )
            self._encoder = None

    # Token counting
    def _count_tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(text) // 4  # rough fallback: ~4 chars per token in English

    # Entry point
    def build(self, reranked_results: list[dict], threads: dict[str, list[dict]]) -> str:
        """
        reranked_results: [{"score": float, "payload": {...}}, ...] — ordered by relevance
        threads:          {thread_id: [chunk_payload, ...]} — output of HybridRetriever.expand_threads()
        """
        if not reranked_results:
            logger.warning("build called with no reranked results — returning empty context")
            return ""

        ordered_thread_ids = self._ordered_unique_thread_ids(reranked_results)

        # First pass: decide which threads fit in budget (need formatted text to count tokens,
        # but final [Email N] numbering must be global, so we number after this pass).
        included_thread_emails = []
        total_tokens = 0
        skipped_threads = 0
        running_count_for_budget_check = 0  # placeholder count, real numbering happens after

        for thread_id in ordered_thread_ids:
            emails = threads.get(thread_id, [])
            if not emails:
                logger.warning("No thread content found for thread_id=%s — skipping", thread_id)
                continue

            # Estimate token cost using placeholder numbering (numbering width barely affects token count)
            preview_block = self._format_thread(emails, start_index=running_count_for_budget_check + 1)
            block_tokens = self._count_tokens(preview_block)

            if total_tokens + block_tokens > self.max_tokens and included_thread_emails:
                skipped_threads += 1
                continue

            included_thread_emails.append(emails)
            total_tokens += block_tokens
            running_count_for_budget_check += len(emails)

        # Second pass: now that we know the final set of included threads, assign
        # globally unique, sequential [Email N of TOTAL] labels across all of them.
        grand_total = sum(len(emails) for emails in included_thread_emails)
        blocks = []
        running_index = 0
        for emails in included_thread_emails:
            block = self._format_thread(emails, start_index=running_index + 1, total_override=grand_total)
            blocks.append(block)
            running_index += len(emails)

        if skipped_threads:
            logger.info(
                "Context budget reached — included %d thread(s)/%d email(s), skipped %d thread(s) | total_tokens=%d/%d",
                len(included_thread_emails), grand_total, skipped_threads, total_tokens, self.max_tokens,
            )
        else:
            logger.info(
                "Context built | threads=%d emails=%d total_tokens=%d/%d",
                len(included_thread_emails), grand_total, total_tokens, self.max_tokens,
            )

        return "\n\n".join(blocks)
    
    # Internal helpers
    def _ordered_unique_thread_ids(self, reranked_results: list[dict]) -> list[str]:
        """Preserve relevance order, drop duplicate thread_ids."""
        seen = set()
        ordered = []
        for r in reranked_results:
            thread_id = r["payload"].get("thread_id", "")
            if thread_id and thread_id not in seen:
                seen.add(thread_id)
                ordered.append(thread_id)
        return ordered

    def _format_thread(self, emails: list[dict], start_index: int, total_override: int | None = None) -> str:
        """
        Emails are expected to already be sorted chronologically (expand_threads does this).
        start_index: the global [Email N] number to start counting from for this thread.
        total_override: the grand total across ALL included threads, for the "of N" part.
                         If None (used only during the budget-estimation pass), falls back
                         to this thread's own length — numbering width doesn't meaningfully
                         affect token count, so the estimate is fine before the real total is known.
        """
        total = total_override if total_override is not None else len(emails)
        blocks = [
            self._format_email(email, start_index + i, total)
            for i, email in enumerate(emails)
        ]
        return "\n\n".join(blocks)

    def _format_email(self, email: dict, index: int, total: int) -> str:
        date   = email.get("date", "")
        sender = email.get("sender_email", "")
        text   = email.get("text", "")
        return (
            f"[Email {index} of {total}] - {date}\n"
            f"From: {sender}\n"
            f"---\n"
            f"{text}"
        )