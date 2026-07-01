"""
Input and output validation guardrails.
InputValidator: basic sanity checks (empty/length) — no injection detection,
this is a single trusted-owner tool, not a public multi-tenant API.
CitationValidator: checks [Email N] references against actual context size.
Tolerant of partial citation failures — only flags as fully invalid when
NONE of the cited emails actually exist in context.
"""

from my_rag_app.constants import CITATION_RE, MAX_QUERY_LENGTH, NO_CONTEXT_MESSAGE
from my_rag_app.entity.reports import ValidationResult
from my_rag_app.logger import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validates user queries before they enter the pipeline."""

    def validate(self, query: str) -> ValidationResult:
        """Validates the user query which can't be empty and greater than max query length"""
        if not query or not query.strip():
            logger.info("Blocked | reason=empty_query")
            return ValidationResult(is_valid=False, reason="Query cannot be empty.")

        if len(query) > MAX_QUERY_LENGTH:
            logger.info("Blocked | reason=query_too_long length=%d", len(query))
            return ValidationResult(is_valid=False, reason=f"Query exceeds {MAX_QUERY_LENGTH} characters.")

        return ValidationResult(is_valid=True)


class CitationValidator:
    """Validates that LLM citations reference real emails in context."""

    def validate(self, response: str, num_context_emails: int) -> ValidationResult:
        """Check that cited [Email N] references exist within the given context size."""
        cited = {int(n) for n in CITATION_RE.findall(response)}

        if not cited:
            # No citations attempted at all — not necessarily wrong (e.g. "I don't know"
            # answers won't cite anything), so this is not treated as invalid.
            return ValidationResult(is_valid=True)

        valid_cited = {n for n in cited if 1 <= n <= num_context_emails}
        invalid_cited = cited - valid_cited

        if invalid_cited:
            logger.warning(
                "Invalid citation(s) found | invalid=%s valid=%s context_size=%d",
                sorted(invalid_cited),
                sorted(valid_cited),
                num_context_emails,
            )

        if not valid_cited:
            # Every citation the model made points outside the actual context —
            # strong signal the answer is not grounded. Treat as invalid.
            return ValidationResult(is_valid=False, reason="No valid citations found in response.")

        return ValidationResult(is_valid=True)

    def fallback_message(self) -> str:
        """Returns the message if no valid answers are found"""
        return NO_CONTEXT_MESSAGE
