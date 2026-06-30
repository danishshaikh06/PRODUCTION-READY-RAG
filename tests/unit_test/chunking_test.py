"""
Unit tests for my_rag_app.core.chunking — pure text-assembly logic, no DB.
Uses lightweight stand-ins instead of real SQLAlchemy model instances,
since _build_text only reads plain attributes off whatever is passed in.
"""

from datetime import datetime
import pytest
from my_rag_app.core.ingestion.chunking import ChunkingPipeline


class FakeEmail:
    def __init__(self, subject, sender_email, recipient_emails, date, body_clean):
        self.subject = subject
        self.sender_email = sender_email
        self.recipient_emails = recipient_emails
        self.date = date
        self.body_clean = body_clean


class FakeMetadata:
    def __init__(self, sender_name="", recipient_names=None):
        self.sender_name = sender_name
        self.recipient_names = recipient_names or []


class TestBuildText:

    def test_includes_recipient_names_for_searchability(self):
        """Reproduces the fix for the 'Mr Trung' retrieval bug: recipient
        display names must appear in the embedded text, not just payload."""
        email = FakeEmail(
            subject="RE: Fuel price for June 2026",
            sender_email="test@smb-freight.com",
            recipient_emails=["test1@skypec.com.vn", "test2@skypec.com.vn"],
            date=datetime(2026, 6, 3, 15, 41, 23),
            body_clean="Referring whatsapp communication, please do the needful on the price.",
        )
        meta = FakeMetadata(
            sender_name="test",
            recipient_names=["test1", "test2"],
        )

        text = ChunkingPipeline()._build_text(email, meta)

        assert "test1" in text
        assert "test" in text
        assert "Referring whatsapp communication" in text

    def test_falls_back_to_email_address_when_sender_name_missing(self):
        email = FakeEmail(
            subject="Test",
            sender_email="mediaf@dca.rak.ae",
            recipient_emails=["slots@smb-freight.com"],
            date=None,
            body_clean="Approved.",
        )
        meta = FakeMetadata(sender_name="", recipient_names=[])

        text = ChunkingPipeline()._build_text(email, meta)

        assert "mediaf@dca.rak.ae" in text

    def test_falls_back_to_raw_emails_when_recipient_names_missing(self):
        email = FakeEmail(
            subject="Test",
            sender_email="a@x.com",
            recipient_emails=["unnamed@x.com"],
            date=None,
            body_clean="Body text.",
        )
        meta = FakeMetadata(sender_name="A", recipient_names=[])

        text = ChunkingPipeline()._build_text(email, meta)

        assert "unnamed@x.com" in text

    def test_chunk_id_is_deterministic_sha256(self):
        pipeline = ChunkingPipeline()
        id1 = pipeline._chunk_id("<test@x.com>")
        id2 = pipeline._chunk_id("<test@x.com>")

        assert id1 == id2
        assert len(id1) == 64  # sha256 hex digest length