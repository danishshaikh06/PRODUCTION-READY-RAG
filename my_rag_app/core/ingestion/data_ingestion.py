"""
Data ingestion — scrapes emails via IMAP, resolves threads (3-tier strategy),
and writes new records directly into the `emails` Postgres table.
Existing email IDs are skipped entirely (immutable once ingested).
"""

import imaplib
import json
import os
import re
import time
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from my_rag_app.config.config import get_session
from my_rag_app.constants import (IMAP_PORT, INGESTION_PROGRESS_FILE,
                                  INGESTION_REPORT_PATH, MAX_RETRIES,
                                  RETRY_DELAY_SECONDS)
from my_rag_app.entity.models import Email
from my_rag_app.logger import get_logger
from my_rag_app.exception.imap_connection import ImapConnectionError, ImapSearchError, ImapConfigError

logger = get_logger(__name__)

REPLY_CHAIN_MARKERS = [
    r"^-{2,}\s*Original Message\s*-{2,}",
    r"^On .{5,100} wrote:$",
]

SUBJECT_PREFIX_RE = re.compile(r"^\s*(re|fw|fwd)\s*:\s*", re.IGNORECASE)


# IMAP fetching
class ImapFetcher:
    """Handles IMAP connection and raw message retrieval, with retry logic."""

    def __init__(
        self, host: str, email_addr: str, password: str, port: int = IMAP_PORT
    ):
        self.host = host
        self.email_addr = email_addr
        self.password = password
        self.port = port
        self.conn = None

    def connect(self) -> None:
        """Connect to the IMAP server with retry logic."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.conn = imaplib.IMAP4_SSL(self.host, self.port)
                self.conn.login(self.email_addr, self.password)
                logger.info("IMAP connected | host=%s attempt=%d", self.host, attempt)
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "IMAP connection attempt %d/%d failed | error=%s",
                    attempt,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        logger.error("IMAP connection failed after %d attempts", MAX_RETRIES)
        raise ImapConnectionError(self.host, last_error)
    
    def fetch_all_message_ids(self, mailbox: str = "INBOX") -> list[bytes]:
        """Return all message IDs in the given mailbox."""
        self.conn.select(mailbox, readonly=True)
        status, data = self.conn.search(None, "ALL")
        if status != "OK":
            logger.error("IMAP search failed | status=%s", status)
            raise ImapSearchError(status)
        ids = data[0].split()
        logger.info("Found %d messages in mailbox '%s'", len(ids), mailbox)
        return ids

    def fetch_raw_message(self, msg_id: bytes) -> bytes | None:
        """Fetch the raw RFC822 bytes for a single message."""
        try:
            status, data = self.conn.fetch(msg_id, "(BODY.PEEK[])")
            if status != "OK" or not data or data[0] is None:
                logger.warning("Fetch failed for msg_id=%s | status=%s", msg_id, status)
                return None
            return data[0][1]
        except Exception as e:
            logger.warning("Exception fetching msg_id=%s | error=%s", msg_id, e)
            return None

    def close(self) -> None:
        """Close the IMAP connection."""
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn.logout()
        except Exception:
            logger.exception("Failed to close/log out IMAP connection")


# Message parsing
class MessageParser:
    """Parses a raw RFC822 message into our intermediate dict format."""

    def parse(self, raw_bytes: bytes) -> dict | None:
        """Parse a raw email message into a structured dictionary."""
        try:
            msg = message_from_bytes(raw_bytes)
        except Exception as e:
            logger.warning("Could not parse raw message | error=%s", e)
            return None

        message_id = self._clean_id(msg.get("Message-ID", ""))
        if not message_id:
            logger.warning("Message has no Message-ID — skipping")
            return None

        subject = self._decode_header_value(msg.get("Subject", ""))
        from_addrs = self._extract_addresses(msg.get("From", ""))
        to_pairs = self._extract_name_email_pairs(msg.get("To", ""))
        to_addrs = [addr for _, addr in to_pairs]
        to_names = [name for name, _ in to_pairs if name]
        date_dt = self._parse_date(msg.get("Date", ""))
        references = self._parse_references(msg.get("References", ""))
        in_reply_to = self._clean_id(msg.get("In-Reply-To", ""))

        body = self._extract_body(msg)
        body = self._strip_reply_chain(body)

        return {
            "id": message_id,
            "subject": subject,
            "body": body,
            "from": from_addrs,
            "to": to_addrs,
            "to_names": to_names,
            "date": date_dt,
            "_references": references,
            "_in_reply_to": in_reply_to,
        }

    def _decode_header_value(self, raw_value: str) -> str:
        if not raw_value:
            return ""
        try:
            parts = decode_header(raw_value)
            decoded = ""
            for text, encoding in parts:
                if isinstance(text, bytes):
                    decoded += text.decode(encoding or "utf-8", errors="replace")
                else:
                    decoded += text
            return decoded.strip()
        except Exception as e:
            logger.debug("Header decode failed | raw=%r error=%s", raw_value, e)
            return raw_value.strip()

    def _clean_id(self, raw_id: str) -> str:
        return raw_id.strip() if raw_id else ""

    def _extract_name_email_pairs(self, raw_value: str) -> list[tuple[str, str]]:
        """Returns [(display_name, email), ...]. display_name is '' if absent.
        getaddresses() splits name/email but does NOT decode MIME encoded-words
        (e.g. '=?utf-8?Q?...?=') in the name part — that must be done separately,
        the same way _decode_header_value() already handles the Subject header."""
        if not raw_value:
            return []
        try:
            pairs = getaddresses([raw_value])
            result = []
            for name, addr in pairs:
                if not addr:
                    continue
                else:
                    decoded_name = self._decode_header_value(name).strip().strip("'\"")
                    result.append((decoded_name, addr.lower().strip()))
            return result
        except Exception as e:
            logger.debug("Address parse failed | raw=%r error=%s", raw_value, e)
            return []

    def _parse_references(self, raw_refs: str) -> list[str]:
        if not raw_refs:
            return []
        return re.findall(r"<[^<>]+>", raw_refs)

    def _extract_addresses(self, raw_value: str) -> list[str]:
        if not raw_value:
            return []
        try:
            pairs = getaddresses([raw_value])
            return [addr.lower().strip() for _, addr in pairs if addr]
        except Exception as e:
            logger.debug("Address parse failed | raw=%r error=%s", raw_value, e)
            return []

    def _parse_date(self, raw_date: str) -> datetime | None:
        if not raw_date:
            return None
        try:
            return parsedate_to_datetime(raw_date)
        except Exception as e:
            logger.debug("Date parse failed | raw=%r error=%s", raw_date, e)
            return None

    def _extract_body(self, msg) -> str:
        plain_text = None
        html_text = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in disposition:
                    continue
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                except Exception as e:
                    logger.debug("Failed to decode part | error=%s", e)
                    continue

                if content_type == "text/plain" and plain_text is None:
                    plain_text = text
                elif content_type == "text/html" and html_text is None:
                    html_text = text
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace") if payload else ""
                if msg.get_content_type() == "text/html":
                    html_text = text
                else:
                    plain_text = text
            except Exception as e:
                logger.debug("Failed to decode single-part body | error=%s", e)

        if plain_text is not None and plain_text.strip():
            return plain_text
        if html_text is not None:
            return self._html_to_text(html_text)
        return ""

    def _html_to_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")
        except Exception as e:
            logger.debug("HTML stripping failed | error=%s", e)
            return html

    def _strip_reply_chain(self, body: str) -> str:
        lines = body.split("\n")
        for i, line in enumerate(lines):
            for pattern in REPLY_CHAIN_MARKERS:
                if re.match(pattern, line.strip()):
                    return "\n".join(lines[:i]).strip()
        return body.strip()


# Thread resolution (3-tier, second pass over all parsed emails)
class ThreadResolver:
    """
    Assigns thread_id and reply_to using a 3-tier strategy:
      1. References header  -> root = first id in the chain
      2. In-Reply-To header -> inherit parent's resolved thread_id
      3. Subject normalization + participant overlap -> fallback grouping
    Tags every record with thread_match_method for provenance.
    """

    def resolve(self, emails: list[dict]) -> list[dict]:
        """Assign thread_id and reply_to using the three-tier resolution strategy."""
        by_id = {e["id"]: e for e in emails}

        for e in emails:
            refs = e.get("_references", [])
            if refs:
                e["thread_id"] = refs[0]
                e["reply_to"] = e.get("_in_reply_to", "") or refs[-1]
                e["thread_match_method"] = "references"

        for e in emails:
            if e.get("thread_id"):
                continue
            parent_id = e.get("_in_reply_to", "")
            if parent_id and parent_id in by_id:
                parent = by_id[parent_id]
                e["thread_id"] = parent.get("thread_id") or parent_id
                e["reply_to"] = parent_id
                e["thread_match_method"] = "in_reply_to"

        unresolved = [e for e in emails if not e.get("thread_id")]
        if unresolved:
            self._resolve_by_subject(unresolved, by_id)

        for e in emails:
            if not e.get("thread_id"):
                e["thread_id"] = e["id"]
                e["reply_to"] = e.get("_in_reply_to", "")
                e["thread_match_method"] = "original"

        return emails

    def _normalize_subject(self, subject: str) -> str:
        prev = None
        s = subject.strip()
        while prev != s:
            prev = s
            s = SUBJECT_PREFIX_RE.sub("", s).strip()
        return s.lower()

    def _resolve_by_subject(self, unresolved: list[dict], by_id: dict) -> None:
        groups: dict[str, list[dict]] = {}
        for e in unresolved:
            norm_subj = self._normalize_subject(e.get("subject", ""))
            if not norm_subj:
                continue
            groups.setdefault(norm_subj, []).append(e)

        for _, group in groups.items():
            if len(group) < 2:
                continue

            linked = self._filter_by_participant_overlap(group)
            if len(linked) < 2:
                continue

            linked.sort(
                key=lambda e: e.get("date") or datetime.min.replace(tzinfo=timezone.utc)
            )
            root = linked[0]
            root_thread_id = root["id"]

            for e in linked:
                e["thread_id"] = root_thread_id
                if e is not root:
                    e["reply_to"] = e.get("_in_reply_to", "") or root_thread_id
                else:
                    e["reply_to"] = e.get("_in_reply_to", "")
                e["thread_match_method"] = "subject_fallback"

    def _filter_by_participant_overlap(self, group: list[dict]) -> list[dict]:
        participant_sets = []
        for e in group:
            participants = set(e.get("from", [])) | set(e.get("to", []))
            participant_sets.append(participants)

        keep = []
        for i, e in enumerate(group):
            has_overlap = any(
                participant_sets[i] & participant_sets[j]
                for j in range(len(group))
                if j != i
            )
            if has_overlap:
                keep.append(e)
        return keep


# Progress tracking (resumable runs)
class ProgressTracker:
    """Tracks which message IDs have already been processed for resumable runs."""
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file

    def load_seen_ids(self) -> set[str]:
        """Return the set of message IDs already processed."""
        if not self.progress_file.exists():
            return set()
        try:
            return set(self.progress_file.read_text(encoding="utf-8").splitlines())
        except Exception as e:
            logger.warning("Could not read progress file | error=%s", e)
            return set()

    def mark_seen(self, msg_id: str) -> None:
        """Record a message ID as processed."""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "a", encoding="utf-8") as f:
                f.write(msg_id + "\n")
        except Exception as e:
            logger.warning("Could not update progress file | error=%s", e)

# Full pipeline
class IngestionPipeline:
    """Scrapes emails via IMAP and writes new records to the database."""
    def __init__(self, mailbox: str = "INBOX"):
        self.mailbox = mailbox
        self.parser = MessageParser()
        self.resolver = ThreadResolver()
        self.progress = ProgressTracker(INGESTION_PROGRESS_FILE)

    def run(self) -> dict:
        """Fetch, resolve threads, and persist new emails end to end."""
        load_dotenv()
        host = os.getenv("IMAP_HOST", "")
        addr = os.getenv("EMAIL_ADDR", "")
        password = os.getenv("EMAIL_PASSWORD", "")

        if not all([host, addr, password]):
            logger.error(
                "Missing IMAP credentials in .env (IMAP_HOST, EMAIL_ADDR, EMAIL_PASSWORD)"
            )
            raise ImapConfigError()

        fetcher = ImapFetcher(host=host, email_addr=addr, password=password)

        parsed_emails: list[dict] = []
        fetch_failures = 0
        parse_failures = 0

        try:
            fetcher.connect()
            msg_ids = fetcher.fetch_all_message_ids(self.mailbox)

            seen_ids = self.progress.load_seen_ids()
            logger.info(
                "Resuming run - %d messages already processed previously", len(seen_ids)
            )

            for idx, msg_id in enumerate(msg_ids, start=1):
                raw = fetcher.fetch_raw_message(msg_id)
                if raw is None:
                    fetch_failures += 1
                    continue

                parsed = self.parser.parse(raw)
                if parsed is None:
                    parse_failures += 1
                    continue

                if parsed["id"] in seen_ids:
                    continue

                parsed_emails.append(parsed)
                self.progress.mark_seen(parsed["id"])

                if idx % 50 == 0:
                    logger.info("Progress: %d/%d messages fetched", idx, len(msg_ids))

        finally:
            fetcher.close()

        logger.info(
            "Fetch complete | fetched=%d fetch_failures=%d parse_failures=%d",
            len(parsed_emails),
            fetch_failures,
            parse_failures,
        )

        # Deduplicate by id within this run (in case IMAP returns the same message twice)
        deduped = {}
        for e in parsed_emails:
            deduped[e["id"]] = e
        parsed_emails = list(deduped.values())

        resolved_emails = self.resolver.resolve(parsed_emails)

        report = self._write_to_db(resolved_emails)
        self._write_report(report)
        return report

    def _write_to_db(self, emails: list[dict]) -> dict:
        method_counts = {
            "references": 0,
            "in_reply_to": 0,
            "subject_fallback": 0,
            "original": 0,
        }
        inserted = 0
        skipped_existing = 0

        with get_session() as session:
            existing_ids = {row[0] for row in session.query(Email.id).all()}

            new_rows = []
            for e in emails:
                if e["id"] in existing_ids:
                    skipped_existing += 1
                    continue

                method_counts[e.get("thread_match_method", "original")] += 1
                new_rows.append(
                    Email(
                        id=e["id"],
                        subject=e["subject"],
                        body_raw=e["body"],
                        body_clean=None,
                        sender_email=(e["from"][0] if e["from"] else ""),
                        recipient_emails=e["to"],
                        recipient_names=e["to_names"],
                        date=e["date"],
                        thread_id=e["thread_id"],
                        reply_to=e["reply_to"] or None,
                        thread_match_method=e["thread_match_method"],
                        is_system_email=False,  # system-email detection happens during cleaning stage
                        cleaned_at=None,
                    )
                )

            if new_rows:
                session.add_all(new_rows)
                session.commit()
                inserted = len(new_rows)

        logger.info(
            "DB write complete | inserted=%d skipped_existing=%d references=%d in_reply_to=%d "
            "subject_fallback=%d original=%d",
            inserted,
            skipped_existing,
            method_counts["references"],
            method_counts["in_reply_to"],
            method_counts["subject_fallback"],
            method_counts["original"],
        )

        return {
            "inserted": inserted,
            "skipped_existing": skipped_existing,
            "thread_match_method_counts": method_counts,
            "total_processed": len(emails),
        }

    def _write_report(self, report: dict) -> None:
        INGESTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(INGESTION_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Ingestion report written | path=%s", INGESTION_REPORT_PATH)
        except Exception as e:
            logger.warning("Could not write ingestion report | error=%s", e)


# Entry point
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
