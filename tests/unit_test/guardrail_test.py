"""
Unit tests for my_rag_app.guardrails — pure logic, no DB/LLM.
"""

from my_rag_app.core.guardrails.pii import PIIDetector
from my_rag_app.core.guardrails.validation import CitationValidator, InputValidator


class TestPIIDetector:
    """Tests for personally identifiable information detection."""

    def setup_method(self):
        """Create a PII detector for each test."""
        self.detector = PIIDetector()

    def test_detects_real_phone_formats(self):
        """All three formats are test examples observed."""
        text = "Call +91-0000000000 or +91 1111111111 or (+91) 22-66852657."
        matches = self.detector.check(text)
        phones = [m.value for m in matches if m.kind == "phone"]

        assert len(phones) == 3

    def test_detects_email_address(self):
        """Ensures email addresses are detected."""
        matches = self.detector.check("Contact test@smb-freight.com for details.")
        emails = [m.value for m in matches if m.kind == "email"]

        assert emails == ["test@smb-freight.com"]

    def test_does_not_flag_request_id_as_phone(self):
        """Regression test: request IDs share digit/dash patterns with phone
        numbers and were originally false-positiving before the fix."""
        matches = self.detector.check("Request ID: LPRQ-07-04-2021-48201")
        phones = [m.value for m in matches if m.kind == "phone"]

        assert phones == []

    def test_does_not_flag_date_as_phone(self):
        """Ensures dates are not misclassified as phone numbers."""
        matches = self.detector.check("The flight departs on 2024-03-08.")
        phones = [m.value for m in matches if m.kind == "phone"]

        assert phones == []

    def test_does_not_flag_flight_number_or_aircraft_reg(self):
        """Ensures flight numbers and aircraft registrations are ignored."""
        matches = self.detector.check("Flight OJ3997 operated by A04-AOA.")

        assert matches == []

    def test_empty_text_returns_no_matches(self):
        """Ensures empty input produces no PII matches."""
        assert self.detector.check("") == []

    def test_clean_text_returns_no_matches(self):
        """Ensures ordinary operational text does not trigger false positives."""
        matches = self.detector.check("The slot was approved with a TOW-BAR precondition.")
        assert matches == []


class TestInputValidator:
    """Tests for user input validation."""

    def setup_method(self):
        """Create an input validator for each test."""
        self.validator = InputValidator()

    def test_rejects_empty_string(self):
        """Ensures empty input is rejected."""
        result = self.validator.validate("")
        assert result.is_valid is False

    def test_rejects_whitespace_only(self):
        """Ensures whitespace-only input is rejected."""
        result = self.validator.validate("   ")
        assert result.is_valid is False

    def test_rejects_overly_long_query(self):
        """Ensures excessively long queries are rejected."""
        result = self.validator.validate("x" * 3000)
        assert result.is_valid is False

    def test_accepts_normal_query(self):
        """Ensures valid user queries are accepted."""
        result = self.validator.validate("Why was the slot for AB3897 approved?")
        assert result.is_valid is True


class TestCitationValidator:
    """Tests for citation validation logic."""

    def setup_method(self):
        """Create a citation validator for each test."""
        self.validator = CitationValidator()

    def test_all_valid_citations_pass(self):
        """Ensures responses with valid citations are accepted."""
        response = "The slot was approved [Email 1] with a condition [Email 2]."
        result = self.validator.validate(response, num_context_emails=3)
        assert result.is_valid is True

    def test_partial_invalid_citations_still_pass(self):
        """Design decision: tolerate partial citation errors rather than
        discarding an otherwise-useful answer over one bad reference number."""
        response = "Per [Email 1] and [Email 7], approved."
        result = self.validator.validate(response, num_context_emails=3)
        assert result.is_valid is True

    def test_all_invalid_citations_fail(self):
        """When every cited email is outside context, the answer is not grounded."""
        response = "Per [Email 9] and [Email 12], approved."
        result = self.validator.validate(response, num_context_emails=3)
        assert result.is_valid is False

    def test_no_citations_attempted_passes(self):
        """An honest 'I don't know' answer with no citations is not a failure."""
        response = "I could not find relevant information to answer this."
        result = self.validator.validate(response, num_context_emails=3)
        assert result.is_valid is True

    def test_fallback_message_is_non_empty(self):
        """Ensures the fallback message is never empty."""
        assert len(self.validator.fallback_message()) > 0
