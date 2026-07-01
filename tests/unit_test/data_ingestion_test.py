"""
Unit tests for my_rag_app.core.ingestion — pure logic only (no IMAP, no DB).
Test cases mirror real bugs found and fixed during development:
  - VAOZ/NOC subject-fallback threading bug
  - MIME-encoded recipient name corruption (the "Trung" bug)
  - Bare "From:" reply-chain stripper destroying forwarded content
"""

from my_rag_app.core.ingestion.data_ingestion import MessageParser, ThreadResolver


# ThreadResolver
class TestThreadResolver:
    """Tests for email thread resolution logic."""

    def test_references_header_resolves_to_shared_root(self):
        """Ensures References headers resolve all replies to the original thread."""
        emails = [
            {
                "id": "<root@x.com>",
                "subject": "Slot Request",
                "from": ["a@x.com"],
                "to": ["b@x.com"],
                "date": None,
                "_references": [],
                "_in_reply_to": "",
            },
            {
                "id": "<reply1@x.com>",
                "subject": "RE: Slot Request",
                "from": ["b@x.com"],
                "to": ["a@x.com"],
                "date": None,
                "_references": ["<root@x.com>"],
                "_in_reply_to": "<root@x.com>",
            },
            {
                "id": "<reply2@x.com>",
                "subject": "RE: RE: Slot Request",
                "from": ["a@x.com"],
                "to": ["b@x.com"],
                "date": None,
                "_references": ["<root@x.com>", "<reply1@x.com>"],
                "_in_reply_to": "<reply1@x.com>",
            },
        ]
        resolved = ThreadResolver().resolve(emails)

        assert resolved[0]["thread_id"] == "<root@x.com>"
        assert resolved[0]["thread_match_method"] == "original"
        assert resolved[1]["thread_id"] == "<root@x.com>"
        assert resolved[1]["thread_match_method"] == "references"
        assert resolved[2]["thread_id"] == "<root@x.com>"
        assert resolved[2]["thread_match_method"] == "references"

    def test_subject_fallback_links_headerless_thread(self):
        """Reproduces the VAOZ/NOC bug: FW:/RE: emails with no References or
        In-Reply-To headers must still be linked via subject + participant overlap."""
        emails = [
            {
                "id": "<orig@smb-freight.com>",
                "subject": "NOC from MoCA to operate cargo flight at VAOZ airport",
                "from": ["test@smb-freight.com"],
                "to": ["test.moca@nic.in"],
                "date": "2026-06-03T11:49:00",
                "_references": [],
                "_in_reply_to": "",
            },
            {
                "id": "<reply@smb-freight.com>",
                "subject": "RE: NOC from MoCA to operate cargo flight at VAOZ airport",
                "from": ["test@gov.in"],
                "to": ["test@smb-freight.com"],
                "date": "2026-06-03T15:30:00",
                "_references": [],
                "_in_reply_to": "",
            },
            {
                "id": "<fwd@smb-freight.com>",
                "subject": "FW: NOC from MoCA to operate cargo flight at VAOZ airport",
                "from": ["test@smb-freight.com"],
                "to": ["test1@smb-freight.com"],
                "date": "2026-06-03T15:36:00",
                "_references": [],
                "_in_reply_to": "",
            },
        ]
        resolved = ThreadResolver().resolve(emails)

        thread_ids = {e["thread_id"] for e in resolved}
        assert len(thread_ids) == 1, "All three NOC emails must share one thread_id"
        assert all(e["thread_match_method"] == "subject_fallback" for e in resolved)

    def test_subject_fallback_does_not_merge_unrelated_participants(self):
        """Same normalized subject, but zero participant overlap — must NOT merge."""
        emails = [
            {
                "id": "<a@x.com>",
                "subject": "RE: Invoice",
                "from": ["alice@x.com"],
                "to": ["bob@x.com"],
                "date": "2026-01-01",
                "_references": [],
                "_in_reply_to": "",
            },
            {
                "id": "<b@x.com>",
                "subject": "RE: Invoice",
                "from": ["carol@y.com"],
                "to": ["dave@y.com"],
                "date": "2026-01-02",
                "_references": [],
                "_in_reply_to": "",
            },
        ]
        resolved = ThreadResolver().resolve(emails)

        assert resolved[0]["thread_id"] != resolved[1]["thread_id"]
        assert all(e["thread_match_method"] == "original" for e in resolved)

    def test_unresolved_email_self_references(self):
        """Ensures standalone emails use their own ID as the thread ID."""
        emails = [
            {
                "id": "<lonely@x.com>",
                "subject": "Unique Subject Nobody Replied To",
                "from": ["a@x.com"],
                "to": ["b@x.com"],
                "date": None,
                "_references": [],
                "_in_reply_to": "",
            },
        ]
        resolved = ThreadResolver().resolve(emails)

        assert resolved[0]["thread_id"] == "<lonely@x.com>"
        assert resolved[0]["thread_match_method"] == "original"


# MessageParser
class TestMessageParser:
    """Tests for email message parsing utilities."""

    def test_extract_name_email_pairs_decodes_mime_encoded_names(self):
        """Reproduces the 'Mr Trung' bug: raw MIME encoded-word names must be
        decoded, not stored as =?utf-8?Q?...?= garbage."""
        parser = MessageParser()
        raw_to = "=?utf-8?Q?'Quang_Trung_Nguy=E1=BB=85n'?= <trungnq@skypec.com.vn>"

        pairs = parser._extract_name_email_pairs(raw_to)

        assert pairs == [("Quang Trung Nguyễn", "trungnq@skypec.com.vn")]

    def test_extract_name_email_pairs_strips_wrapping_quotes(self):
        """Ensures surrounding quotes are removed from display names."""
        parser = MessageParser()
        raw_to = "'Manager Pham Cong Dien Sales' <dienpc@skypec.com.vn>"

        pairs = parser._extract_name_email_pairs(raw_to)

        assert pairs == [("Manager Pham Cong Dien Sales", "dienpc@skypec.com.vn")]

    def test_extract_name_email_pairs_handles_multiple_recipients(self):
        """Ensures multiple recipients are parsed correctly."""
        parser = MessageParser()
        raw_to = "Alice <alice@x.com>, Bob <bob@x.com>"

        pairs = parser._extract_name_email_pairs(raw_to)

        assert pairs == [("Alice", "alice@x.com"), ("Bob", "bob@x.com")]

    def test_extract_name_email_pairs_empty_input(self):
        """Ensures empty recipient strings return an empty list."""
        parser = MessageParser()
        assert parser._extract_name_email_pairs("") == []

    def test_strip_reply_chain_preserves_forward_header_content(self):
        """Reproduces the bare-From: bug: a forwarded email's inline
        From:/Sent:/To:/Subject: block must NOT be treated as a quote marker
        and truncate real content."""
        parser = MessageParser()
        body = (
            "Dear Naupada Satyanarayana Ji,\n\n"
            "We are planning to operate cargo flights from Nashik Airport.\n\n"
            "From: someone@moca.gov.in\n"
            "Sent: Tuesday, June 2, 2026\n"
            "To: test@smb-freight.com\n"
            "Subject: Original NOC request\n\n"
            "Many thanks for your support."
        )
        result = parser._strip_reply_chain(body)

        assert "Many thanks for your support." in result
        assert "Nashik Airport" in result

    def test_strip_reply_chain_removes_original_message_marker(self):
        """Ensures quoted content after an Original Message marker is removed."""
        parser = MessageParser()
        body = "Sure, that works.\n\n-----Original Message-----\nFrom: x@x.com\nOld quoted content."

        result = parser._strip_reply_chain(body)

        assert result == "Sure, that works."
        assert "Old quoted content" not in result

    def test_strip_reply_chain_removes_on_wrote_marker(self):
        """Ensures quoted content after an 'On ... wrote:' marker is removed."""
        parser = MessageParser()
        body = "Thanks, confirmed.\n\nOn Tue, Jun 3, 2026 at 10:00 AM, x@x.com wrote:\nOld quoted content."

        result = parser._strip_reply_chain(body)

        assert result == "Thanks, confirmed."
        assert "Old quoted content" not in result
