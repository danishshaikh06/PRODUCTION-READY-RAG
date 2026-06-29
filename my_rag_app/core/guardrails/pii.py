"""
PII detection for LLM-generated answers — detect and log only, no masking.
This is a single-user trusted-owner tool; the purpose is visibility into
how often contact info surfaces in answers, not protecting the user from
their own company's data.
"""

import re
from my_rag_app.logger import get_logger
from my_rag_app.constants import PHONE_RE, EMAIL_RE
from my_rag_app.entity.reports import PIIMatch

logger = get_logger(__name__)

class PIIDetector:

    def check(self, text: str) -> list[PIIMatch]:
        if not text:
            return []

        matches = []
        for m in EMAIL_RE.finditer(text):
            matches.append(PIIMatch(kind="email", value=m.group()))
        for m in PHONE_RE.finditer(text):
            # Skip short numeric noise (e.g. "2024", flight numbers) — require
            # at least 7 digits total to count as a plausible phone number.
            digits = re.sub(r"\D", "", m.group())
            if len(digits) >= 7:
                matches.append(PIIMatch(kind="phone", value=m.group()))

        if matches:
            logger.info("PII detected | count=%d kinds=%s", len(matches), [m.kind for m in matches])

        return matches