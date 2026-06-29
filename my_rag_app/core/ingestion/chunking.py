"""
Chunking — joins cleaned emails with their metadata, builds the embedding
text (Subject + From + To + Date + Body), and writes one chunk per email
into the `chunks` table ("1 email = 1 chunk").
"""

import json
import hashlib
from datetime import datetime, timezone

from my_rag_app.logger import get_logger
from my_rag_app.entity.models import Email, Metadata, Chunk
from my_rag_app.entity.reports import ChunkingReport
from my_rag_app.config.config import get_session
from my_rag_app.constants import CHUNKING_REPORT_PATH

logger = get_logger(__name__)


class ChunkingPipeline:

    def run(self) -> ChunkingReport:
        with get_session() as session:
            chunked_ids = {row[0] for row in session.query(Chunk.email_id).all()}
            rows = (
                session.query(Email, Metadata)
                .join(Metadata, Metadata.email_id == Email.id)
                .filter(Email.body_clean.isnot(None))
                .filter(Email.is_system_email.is_(False))
                .filter(~Email.id.in_(chunked_ids))
                .all()
            )
            logger.info("Chunking | %d email(s) pending", len(rows))

            created = skipped_empty = 0
            for email, meta in rows:
                if not email.body_clean.strip():
                    skipped_empty += 1
                    continue

                text = self._build_text(email, meta)
                session.add(
                    Chunk(
                        chunk_id=self._chunk_id(email.id),
                        email_id=email.id,
                        thread_id=email.thread_id,
                        text=text,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                created += 1

            session.commit()

        report = ChunkingReport(chunks_created=created, skipped_empty_body=skipped_empty)
        logger.info("Chunking complete | %s", report)
        self._write_report(report)
        return report

    def _build_text(self, email: Email, meta: Metadata) -> str:
        sender = f"{meta.sender_name} <{email.sender_email}>" if meta.sender_name else email.sender_email

        if meta.recipient_names:
            recipient = f"{', '.join(meta.recipient_names)} <{', '.join(email.recipient_emails)}>"
        else:
            recipient = ", ".join(email.recipient_emails)

        return (
            f"Subject: {email.subject}\n"
            f"From: {sender}\n"
            f"To: {recipient}\n"
            f"Date: {email.date}\n"
            f"Body: {email.body_clean}"
        )

    def _chunk_id(self, email_id: str) -> str:
        return hashlib.sha256(email_id.encode("utf-8")).hexdigest()

    def _write_report(self, report: ChunkingReport) -> None:
        CHUNKING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKING_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report.__dict__, f, indent=2)


if __name__ == "__main__":
    ChunkingPipeline().run()