from dataclasses import dataclass, field


@dataclass
class IngestionReport:
    """Summary of a single ingestion pipeline run."""

    inserted: int
    skipped_existing: int
    total_processed: int
    thread_match_method_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class CleaningReport:
    """Summary of a single cleaning pipeline run."""

    cleaned: int
    system_emails: int
    empty_after_clean: int


@dataclass
class ChunkingReport:
    """Summary of a single chunking pipeline run."""

    chunks_created: int
    skipped_empty_body: int


@dataclass
class MetadataReport:
    """Summary of a single metadata extraction run."""

    extracted: int


@dataclass
class QdrantIngestionReport:
    """Summary of a single Qdrant embedding and upsert run."""

    chunks_pending: int
    points_upserted: int
    skipped_empty: int
    batches_failed: int


@dataclass
class PIIMatch:
    """A single detected piece of PII (kind and value)."""

    kind: str  # "phone" or "email"
    value: str


@dataclass
class ValidationResult:
    """Result of an input or output validation check."""

    is_valid: bool
    reason: str = ""


@dataclass
class LLMResponse:
    """Response returned from the LLM client, including token and latency stats."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
