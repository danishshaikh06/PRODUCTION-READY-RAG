"""
Data cleaning — loads emails with body_clean IS NULL, strips noise from the
body, flags system emails, and writes results back to the same rows.
"""

import re
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

from my_rag_app.entity.reports import CleaningReport
from my_rag_app.logger import get_logger
from my_rag_app.entity.models import Email
from my_rag_app.config.config import get_session
from my_rag_app.constants import CLEANING_REPORT_PATH 

logger = get_logger(__name__)

# Patterns
GREETING_RE = re.compile(r"^Dear\s+.{1,60}?\s*,?\s*$", re.MULTILINE)
SIGNATURE_ANCHOR_RE = re.compile(
    r"(Thanks\s*&\s*Regards|Best\s+[Rr]egards|Regards\s*,?|Sincerely\s*,?).*",
    re.DOTALL | re.IGNORECASE,
)
CONFIDENTIALITY_RE = re.compile(r"This\s+E[\-\s]?Mail\s+and\s+any\s+files\s+transmitted.*", re.DOTALL | re.IGNORECASE)
PRINT_REMINDER_RE  = re.compile(r"We\s+have\s+a\s+responsibility\s+to\s+the\s+environment.*", re.DOTALL | re.IGNORECASE)
CID_RE             = re.compile(r"\[cid:[^\]]+\]", re.IGNORECASE)
OUTLOOK_FOOTER_RE  = re.compile(r"Get\s+Outlook\s+for\s+\w+.*", re.DOTALL | re.IGNORECASE)
SENT_FROM_RE       = re.compile(r"Sent\s+from\s+my\s+\w[\w\s]{0,20}", re.IGNORECASE)
SOCIAL_MEDIA_RE    = re.compile(r"^.*(linkedin|twitter|facebook|instagram|youtube).*$", re.MULTILINE | re.IGNORECASE)
MARKETING_RE       = re.compile(r"Asia.s\s+Youngest\s+Aircraft\s+Fleet.*", re.DOTALL | re.IGNORECASE)
ENCODING_RE        = re.compile(r"Â\xa0|Â |Â")
FEEDBACK_RE        = re.compile(
    r"(To\s+serve\s+you\s+better|please\s+complete\s+this\s+survey|click\s+here\s+for\s+valuable\s+feedback|provide\s+your\s+valuable\s+feedback).*",
    re.DOTALL | re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+")

SYSTEM_SENDER_DOMAINS   = {"ionos.com", "mailer-daemon"}
SYSTEM_SUBJECT_PREFIXES = ("welcome to mail", "daily report mailbox", "spam report")


class CleaningPipeline:

    def run(self) -> dict:
        with get_session() as session:
            rows = session.query(Email).filter(Email.body_clean.is_(None)).all()
            logger.info("Cleaning | %d email(s) pending", len(rows))

            system_count = empty_count = 0
            for row in rows:
                row.is_system_email = self._is_system_email(row.sender_email, row.subject)
                row.body_clean = "" if row.is_system_email else self._clean_body(row.body_raw)
                row.cleaned_at = datetime.now(timezone.utc)

                if row.is_system_email:
                    system_count += 1
                elif not row.body_clean.strip():
                    empty_count += 1

            session.commit()

        data_cleaning_report = CleaningReport(cleaned= len(rows), system_emails= system_count, empty_after_clean= empty_count)
        report = asdict(data_cleaning_report)
        logger.info("Cleaning complete | %s", report)
        self._write_report(report)
        return report

    def _is_system_email(self, sender_email: str, subject: str) -> bool:
        domain = sender_email.split("@")[-1].lower() if "@" in sender_email else ""
        subject_lower = (subject or "").lower()
        return domain in SYSTEM_SENDER_DOMAINS or any(subject_lower.startswith(p) for p in SYSTEM_SUBJECT_PREFIXES)

    def _clean_body(self, body: str) -> str:
        body = ENCODING_RE.sub(" ", body or "")
        for pattern in (CID_RE, CONFIDENTIALITY_RE, PRINT_REMINDER_RE, FEEDBACK_RE,
                        MARKETING_RE, SOCIAL_MEDIA_RE, OUTLOOK_FOOTER_RE, SENT_FROM_RE, URL_RE):
            body = pattern.sub("", body)
        body = GREETING_RE.sub("", body)
        body = SIGNATURE_ANCHOR_RE.sub("", body)
        return self._normalize_whitespace(body)

    def _normalize_whitespace(self, text: str) -> str:
        lines = [l.strip() for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        cleaned, prev_blank = [], False
        for line in lines:
            is_blank = line == ""
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank
        return "\n".join(cleaned).strip()

    def _write_report(self, report: dict) -> None:
        CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLEANING_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    CleaningPipeline().run()