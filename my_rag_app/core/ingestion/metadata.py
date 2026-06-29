"""
Metadata extraction — derives sender identity (name/company/designation),
recipient names, and greeting-line name from cleaned emails, writing one
row per email into the `metadata` table.
"""

import re
import json
from pathlib import Path
from datetime import datetime, timezone

from my_rag_app.logger import get_logger
from my_rag_app.entity.models import Email, Metadata
from my_rag_app.entity.reports import MetadataReport
from my_rag_app.config.config import get_session
from my_rag_app.constants import METADATA_REPORT_PATH

logger = get_logger(__name__)


# Patterns
GREETING_CAPTURE_RE = re.compile(r"^Dear\s+(.+?)\s*,", re.MULTILINE)

SIGNATURE_ANCHOR_RE = re.compile(
    r"(Thanks\s*&\s*Regards|Best\s+[Rr]egards|Regards\s*,?|Sincerely\s*,?)",
    re.IGNORECASE,
)

DESIGNATION_KEYWORDS = [
    "executive", "officer", "manager", "director", "engineer",
    "coordinator", "supervisor", "head", "senior", "junior",
    "operations", "assistant", "analyst", "controller",
]

KNOWN_COMPANIES = sorted([
    "Mumbai International Airport Pvt Ltd", "Mumbai International Airport Ltd",
    "Mumbai International Airport", "Airport Operations Control Centre",
    "SMB Freight FZE", "SMB Freight", "SMB-F",
    "Department of Civil Aviation", "Adani Airports", "Adani Airport",
    "RAKDCA", "RAK DCA", "SalamAir", "Omega Air", "AOCC",
], key=len, reverse=True)


class MetadataPipeline:

    def run(self) -> MetadataReport:
        with get_session() as session:
            extracted_ids = {row[0] for row in session.query(Metadata.email_id).all()}
            rows = (
                session.query(Email)
                .filter(Email.body_clean.isnot(None))
                .filter(~Email.id.in_(extracted_ids))
                .all()
            )
            logger.info("Metadata extraction | %d email(s) pending", len(rows))

            for row in rows:
                name, company, designation = self._extract_signature_fields(row.body_raw)
                greeting_name = self._extract_greeting_name(row.body_raw)

                session.add(
                    Metadata(
                        email_id=row.id,
                        sender_name=name,
                        sender_company=company,
                        sender_designation=designation,
                        recipient_names=row.recipient_names,
                        greeting_name=greeting_name,
                        extracted_at=datetime.now(timezone.utc),
                    )
                )

            session.commit()

        report = MetadataReport(extracted=len(rows))
        logger.info("Metadata extraction complete | %s", report)
        self._write_report(report)
        return report

    def _extract_greeting_name(self, body_raw: str) -> str:
        match = GREETING_CAPTURE_RE.search(body_raw or "")
        return match.group(1).strip() if match else ""

    def _extract_signature_fields(self, body: str) -> tuple[str, str, str]:
        match = SIGNATURE_ANCHOR_RE.search(body or "")
        if not match:
            return "", "", ""

        after = body[match.start():]
        lines = [l.strip() for l in after.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        content_lines = [l for l in lines[1:] if l][:8]
        if not content_lines:
            return "", "", ""

        name = ""
        candidate = content_lines[0]
        words = candidate.split()
        if 2 <= len(words) <= 4 and not candidate.isupper() and not any(
            kw in candidate.lower() for kw in DESIGNATION_KEYWORDS
        ):
            name = candidate

        designation = company = ""
        for line in content_lines[1:]:
            if not designation and any(kw in line.lower() for kw in DESIGNATION_KEYWORDS) and len(line.split()) <= 6:
                designation = line
            if not company:
                for known in KNOWN_COMPANIES:
                    if known.lower() in line.lower():
                        company = known
                        break
            if designation and company:
                break

        return name, company, designation

    def _write_report(self, report: MetadataReport) -> None:
        METADATA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report.__dict__, f, indent=2)


if __name__ == "__main__":
    MetadataPipeline().run()