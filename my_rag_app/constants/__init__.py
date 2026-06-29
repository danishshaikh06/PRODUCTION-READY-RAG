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
"""
Centralized constants for the RAG pipeline.
Single flat module — import everything from `my_rag_app.constants`.
"""
from pathlib import Path


# Database table names
EMAILS_TABLE   = "emails"
METADATA_TABLE = "metadata"
CHUNKS_TABLE   = "chunks"


# Qdrant configuration
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "email_knowledge_v1"
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE        = 384          # BAAI/bge-small-en-v1.5 dimension
DISTANCE_METRIC    = "cosine"



# Chunking strategy
# 1 email = 1 chunk. No size/overlap settings — splitting would break the
# semantic unit of an email (approvals, conditions, decisions live in the
# whole message, not an arbitrary window of it).
CHUNKING_STRATEGY = "email_chunk"


# Model names
DENSE_EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DENSE_DIM              = 384
RERANKER_MODEL         = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL              = "qwen2.5:1.5b"
LLM_BASE_URL           = "http://localhost:11434"
TOKENIZER_ENCODING     = "cl100k_base"


# Pipeline configs-
MAX_RETRIES            = 3
RETRY_DELAY_SECONDS    = 5
EMBEDDING_BATCH_SIZE   = 64
LLM_REQUEST_TIMEOUT_SECONDS = 120

DEFAULT_TOP_K_RETRIEVE = 10
DEFAULT_TOP_K_RERANK   = 3
CONTEXT_MAX_TOKENS     = 5000



# General-purpose regex (multi-use only — single-use cleaning patterns stay
# local to data_cleaning.py, not duplicated here)
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
URL_REGEX   = r"https?://\S+"

# Artifact report paths — one central place, so renaming a path means editing one line
ARTIFACTS_DIR           = Path("artifacts")
INGESTION_REPORT_PATH   = ARTIFACTS_DIR / "raw" / "ingestion_report.json"
CLEANING_REPORT_PATH    = ARTIFACTS_DIR / "cleaned" / "cleaning_report.json"
CHUNKING_REPORT_PATH    = ARTIFACTS_DIR / "chunks" / "chunking_report.json"
METADATA_REPORT_PATH    = ARTIFACTS_DIR / "metadata" / "metadata_report.json"
INGESTION_PROGRESS_FILE = ARTIFACTS_DIR / "raw"/".scrape_progress"
CLEANING_PROGRESS_FILE = ARTIFACTS_DIR / "cleaned"/".scrape_progress"
METADATA_PROGRESS_FILE = ARTIFACTS_DIR / "metadata"/".scrape_progress"




# System prompt (RAG-specific) — moved from prompting/builder.py for
# centralized access; builder.py imports this rather than redefining it.
PROMPT_VERSION = "v1"

SYSTEM_PROMPT_V1 = """You are an Email Intelligence Assistant for SMB Freight FZE aviation operations. Never show reasoning steps, hidden thoughts, or analysis.

RULES:
1. Answer ONLY using the provided email context. Do not use outside knowledge.
2. If the context does not contain enough information, say so explicitly.
3. Cite the specific email(s) that support your answer using [Email N] format, where N matches the numbering in the context.
4. For timeline/lifecycle questions, present information chronologically.
5. Preserve exact operational details (flight numbers, times, conditions, NOTAM references, request IDs).
6. Do not reveal personal contact information (phone numbers, personal emails) unless specifically asked.
7. If multiple emails conflict (e.g. a later email revises an earlier decision), prefer the most recent one and note the change."""

# GuardRails 
#PII 
PHONE_RE = (r"\(?\+\d{1,3}\)?[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
EMAIL_RE = (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

#Validation
MAX_QUERY_LENGTH = 2000
CITATION_RE = (r"\[Email (\d+)\]")
NO_CONTEXT_MESSAGE = "I couldn't find relevant information in the emails to answer this."