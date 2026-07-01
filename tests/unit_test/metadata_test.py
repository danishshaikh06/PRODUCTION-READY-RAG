"""
Unit tests for my_rag_app.core.metadata — pure regex/text extraction, no DB.
"""

from my_rag_app.core.ingestion.metadata import MetadataPipeline


class TestGreetingNameExtraction:
    """Tests for extracting greeting names from email bodies."""

    def setup_method(self):
        """Create a metadata pipeline for each test."""
        self.pipeline = MetadataPipeline()

    def test_dear_name_with_trailing_space_before_comma(self):
        """Ensures names with spaces before commas are extracted correctly."""
        body = "Dear Ibrahim Shaikh ,\n\nBelow is the result..."
        assert self.pipeline._extract_greeting_name(body) == "Ibrahim Shaikh"

    def test_dear_sir_madam(self):
        """Ensures 'Sir/Madam' greetings are extracted correctly."""
        body = "Dear Sir/Madam,\n\nYA copy received."
        assert self.pipeline._extract_greeting_name(body) == "Sir/Madam"

    def test_dear_team_name(self):
        """Ensures team names are extracted from greetings."""
        body = "Dear AOCC Team,\n    \n    Certainly, as discussed..."
        assert self.pipeline._extract_greeting_name(body) == "AOCC Team"

    def test_dear_multi_word_name_with_title(self):
        """Ensures multi-word names with titles are extracted correctly."""
        body = "Dear Naupada Satyanarayana Ji,\n\n\nGreetings from SMB Freight!"
        assert self.pipeline._extract_greeting_name(body) == "Naupada Satyanarayana Ji"

    def test_dear_all(self):
        """Ensures 'All' greetings are extracted correctly."""
        body = "Dear All,\n\nThe below mentioned flight is approved."
        assert self.pipeline._extract_greeting_name(body) == "All"

    def test_dear_acronym(self):
        """Ensures acronym greetings are extracted correctly."""
        body = "Dear SOD,\n\nThe slot timings..."
        assert self.pipeline._extract_greeting_name(body) == "SOD"

    def test_no_greeting_returns_empty(self):
        """Ensures missing greetings return an empty string."""
        body = "Slot timings noted. No greeting line here."
        assert self.pipeline._extract_greeting_name(body) == ""


class TestSignatureFieldExtraction:
    """Tests for extracting signature fields from email bodies."""

    def setup_method(self):
        """Create a metadata pipeline for each test."""
        self.pipeline = MetadataPipeline()

    def test_extracts_name_company_designation(self):
        """Ensures signature name, company, and designation are extracted."""
        body = (
            "Dear AOCC Team,\n\n"
            "Certainly, as discussed during our call, team, I kindly request "
            "your confirmation for the slot on 9th March.\n\n"
            "Regards,\n"
            "Danish Shaikh\n"
            "Operation Executive\n"
            "SMB-F\n"
            "(+91-0000000000)"
        )
        name, company, _ = self.pipeline._extract_signature_fields(body)

        assert name == "Danish Shaikh"
        assert company == "SMB-F"
        assert _ == "Operation Executive"

    def test_no_signature_anchor_returns_empty_tuple(self):
        """Ensures missing signatures return empty extracted fields."""
        body = "Just a body with no closing salutation at all."
        name, company, designation = self.pipeline._extract_signature_fields(body)

        assert (name, company, designation) == ("", "", "")

    def test_rejects_allcaps_designation_line_as_name(self):
        """A department name in all caps should not be mistaken for a person's name."""
        body = "Approved.\n\nThanks & Regards\n\nAIRPORT OPERATIONS CONTROL CENTRE\nMumbai International Airport Ltd"
        name, company, _ = self.pipeline._extract_signature_fields(body)

        assert name == ""  # all-caps line correctly rejected as a name candidate
        assert company == "Mumbai International Airport Ltd"
