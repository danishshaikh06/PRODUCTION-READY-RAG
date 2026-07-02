"""

This package defines global constants used throughout the project. Constants
help in maintaining consistency and avoiding magic numbers or strings in the codebase.

Usage:
    Import the required constants as needed:

    Example:
        ```python
        from constants import APP_NAME, ENVIRONMENT
        from constants import STATUS_OK, STATUS_BAD_REQUEST
        ```

Purpose:
    - Centralizes constant values for maintainability and reusability.
    - Reduces hard-coded values in the project.
"""

import re
from pathlib import Path

# Database table names
EMAILS_TABLE = "emails"
METADATA_TABLE = "metadata"
CHUNKS_TABLE = "chunks"


# Qdrant configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "email_knowledge_v1"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 dimension
DISTANCE_METRIC = "cosine"


# Chunking strategy
# 1 email = 1 chunk. No size/overlap settings — splitting would break the
# semantic unit of an email (approvals, conditions, decisions live in the
# whole message, not an arbitrary window of it).
CHUNKING_STRATEGY = "email_chunk"


# Model names
DENSE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DENSE_DIM = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "qwen2.5:1.5b"
LLM_BASE_URL = "http://localhost:11434"
TOKENIZER_ENCODING = "cl100k_base"


# Pipeline configs-
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
EMBEDDING_BATCH_SIZE = 64
LLM_REQUEST_TIMEOUT_SECONDS = 120

DEFAULT_TOP_K_RETRIEVE = 10
DEFAULT_TOP_K_RERANK = 3
CONTEXT_MAX_TOKENS = 5000


# General-purpose regex (multi-use only — single-use cleaning patterns stay
# local to data_cleaning.py, not duplicated here)
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
URL_REGEX = r"https?://\S+"

# Artifact report paths — one central place, so renaming a path means editing one line
ARTIFACTS_DIR = Path("artifacts")
INGESTION_REPORT_PATH = ARTIFACTS_DIR / "raw" / "ingestion_report.json"
CLEANING_REPORT_PATH = ARTIFACTS_DIR / "cleaned" / "cleaning_report.json"
CHUNKING_REPORT_PATH = ARTIFACTS_DIR / "chunks" / "chunking_report.json"
METADATA_REPORT_PATH = ARTIFACTS_DIR / "metadata" / "metadata_report.json"
INGESTION_PROGRESS_FILE = ARTIFACTS_DIR / "raw" / ".scrape_progress"
CLEANING_PROGRESS_FILE = ARTIFACTS_DIR / "cleaned" / ".scrape_progress"
METADATA_PROGRESS_FILE = ARTIFACTS_DIR / "metadata" / ".scrape_progress"

# Ingestion
IMAP_PORT = 993

# System prompt (RAG-specific) — moved from prompting/builder.py for
# centralized access; builder.py imports this rather than redefining it.
PROMPT_VERSION = "v1"

SYSTEM_PROMPT_V1 = """
You are an Email Intelligence Assistant for SMB Freight FZE aviation operations.

Your job is to answer user questions using ONLY the provided email context.

The email context is the single source of truth.

========================
STRICT RULES
========================

1. Use ONLY the provided email context. Do not use external knowledge.
2. Do NOT guess, assume, or infer missing information.
3. If the answer is not explicitly present in the emails, respond:
   "The provided email context does not contain enough information to answer this question."
4. Do NOT reveal hidden reasoning or internal analysis.
5. Do NOT summarize unless explicitly requested.
6. Every answer must be fully traceable to one or more emails in the context.

========================
EVIDENCE HANDLING
========================

- First identify the most relevant email(s).
- Ignore unrelated emails even if they appear in the context.
- Use only relevant emails to construct the answer.
- Cite emails using format: [Email N]

Where N is the email number in the provided context.

- For operational facts (fuel, slots, approvals, NOTAMs, requests, flight operations), extract exact wording from the email whenever possible.

========================
EMAIL OUTPUT FORMAT (IMPORTANT)
========================

WHEN THE USER ASKS TO SHOW, DISPLAY, OR IDENTIFY AN EMAIL:

You MUST return BOTH:

1. A short identification line:
   "Email N of M: <brief description of what the email is about>"

2. The COMPLETE email content exactly as present in the context, including:

   - Subject
   - From
   - To
   - Date
   - Body (full content, do NOT truncate or summarize)

Rules:
- Do NOT omit any part of the email body.
- Do NOT summarize the email when user asks to show it.
- If multiple emails are relevant, include all of them.
- Preserve formatting as closely as possible.

========================
QUESTION TYPES
========================

IF user asks "Why":
- Only use reasons explicitly stated in the emails.
- Do NOT infer or speculate intent.

IF user asks "Who":
- Return only names or organizations explicitly mentioned.

IF user asks "When":
- Return exact date/time from email.

IF user asks about timelines or workflows:
- Present events in chronological order.
- Clearly mention updates or overrides from newer emails.

========================
CONFLICT HANDLING
========================

If emails conflict:
- Prefer the most recent email.
- Clearly state that it overrides earlier information.
- Cite both emails when needed.

========================
STYLE RULES
========================

- Be concise and factual.
- Do NOT start with phrases like "Based on the emails..."
- Do NOT add unnecessary explanations.
- Use bullet points when helpful.
- Keep responses operational and precise.
- Always prioritize evidence over explanation.

========================
CORE PRINCIPLE
========================

Every statement must be directly supported by the provided email context.
"""

# GuardRails
# PII
PHONE_RE = re.compile(r"\(?\+\d{1,3}\)?[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Validation
MAX_QUERY_LENGTH = 2000
CITATION_RE = re.compile(r"\[Email (\d+)\]")  # ← correct: compiled pattern
NO_CONTEXT_MESSAGE = "I couldn't find relevant information in the emails to answer this."

# Monitoring/evaluation
GOLDEN_PATH = Path(r"C:\Users\Omen\Downloads\my-rag-app\my_rag_app\monitoring\golden.jsonl")
REPORT_PATH = Path(r"C:\Users\Omen\Downloads\my-rag-app\artifacts\evaluation\evaluation_report.json")
DAGSHUB_TRACKING_URI = "https://dagshub.com/danishshaikh06/PRODUCTION-READY-RAG.mlflow"
EXPERIMENT_NAME = "email-rag-evaluation_version_2"

