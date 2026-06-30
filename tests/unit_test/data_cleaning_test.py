"""
Unit tests for my_rag_app.core.data_cleaning — pure regex/text logic, no DB.
"""
import pytest 
from my_rag_app.core.ingestion.data_cleaning import CleaningPipeline

class TestCleaningPipeline:

    def test_clean_body_preserves_operational_content_strips_signature(self):
        """The original AOCC test case: TOW-BAR condition must survive,
        the full corporate signature block must not."""
        body = (
            "Dear Sir/Madam,\n\n"
            "YA copy received.\n\n"
            "This slot is approved with a precondition of having serviceable "
            "TOW-BAR on board OR with ground handling agency.\n\n"
            "Thanks & Regards\n\n"
            "Tanisha Sawant\n\n"
            "Officer\n\n"
            "AIRPORT OPERATIONS CONTROL CENTRE\n"
            "Mumbai International Airport Pvt Ltd\n"
            "Direct Line:  +91 22 66852550"
        )
        cleaned = CleaningPipeline()._clean_body(body)

        assert "TOW-BAR" in cleaned
        assert "YA copy received" in cleaned
        assert "Tanisha" not in cleaned
        assert "Direct Line" not in cleaned

    def test_clean_body_strips_confidentiality_block(self):
        body = (
            "Landing permission APPROVED for non schedule flt cargo ops.\n\n"
            "Best regards,\n\n"
            "Department of Civil Aviation\n\n"
            "This E-Mail and any files transmitted with it are confidential "
            "and intended solely for the use of the individual..."
        )
        cleaned = CleaningPipeline()._clean_body(body)

        assert "APPROVED" in cleaned
        assert "confidential" not in cleaned

    def test_clean_body_strips_print_reminder(self):
        body = "Approved.\n\nBest regards,\n\nDCA\n\nWe have a responsibility to the environment. So let us please think before we print."
        cleaned = CleaningPipeline()._clean_body(body)

        assert "responsibility to the environment" not in cleaned

    def test_clean_body_strips_cid_references(self):
        body = "See attached image [cid:c01e7193-1bc6-484c-bc9f-c537346061e6] for details."
        cleaned = CleaningPipeline()._clean_body(body)

        assert "cid:" not in cleaned
        assert "for details" in cleaned

    def test_clean_body_fixes_encoding_artifacts(self):
        body = "Operation ExecutiveÂ \nSMB-FÂ \n(+91-0000000000)Â"
        cleaned = CleaningPipeline()._clean_body(body)

        assert "Â" not in cleaned

    def test_clean_body_handles_empty_string(self):
        assert CleaningPipeline()._clean_body("") == ""

    def test_is_system_email_detects_ionos_domain(self):
        pipeline = CleaningPipeline()
        assert pipeline._is_system_email("support@ionos.com", "Welcome to Mail Basic") is True

    def test_is_system_email_false_for_real_sender(self):
        pipeline = CleaningPipeline()
        assert pipeline._is_system_email("aocc.planning@adani.com", "RE: Slot Request") is False

    def test_normalize_whitespace_collapses_blank_lines(self):
        text = "Line one\n\n\n\n\nLine two\r\n\r\n\r\nLine three"
        result = CleaningPipeline()._normalize_whitespace(text)

        assert "\n\n\n" not in result
        assert "Line one" in result and "Line two" in result and "Line three" in result