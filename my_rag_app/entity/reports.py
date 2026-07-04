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


# LLM response dataclass
@dataclass
class LLMResponse:
    """Response returned from the LLM client, including token and latency stats."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


# Monitoring/evaluation report dataclasses
# Data classes
@dataclass
class GoldenQuery:
    """A single golden evaluation query with expected outputs."""

    query: str
    expected_email_ids: list[str]
    expected_answer_contains: list[str]
    notes: str = ""


@dataclass
class GoldenQuery_V2:
    """A single golden evaluation query with expected outputs."""

    query: str
    ground_truth_context: list[str]
    ground_truth_answer: str
    key_facts: list[str]
    metadata: dict


@dataclass
class QueryResult:
    """Evaluation result for a single golden query."""

    query: str
    retrieved_email_ids: list[str]
    answer: str
    retrieval_recall: float
    answer_quality: float
    latency_ms: float
    notes: str = ""


@dataclass
class QueryResult_v2:
    """Evaluation result for a single golden query."""

    query: str
    retrieved_email_ids: list[str]
    answer: str
    retrieval_recall: float
    answer_quality_semantic: float
    latency_ms: float
    fact_f1: float


@dataclass
class EvaluationReport:
    """Aggregated results across all golden queries for one evaluation run."""

    total_queries: int = 0
    avg_retrieval_recall: float = 0.0
    avg_answer_quality: float = 0.0
    avg_latency_ms: float = 0.0
    query_results: list[QueryResult] = field(default_factory=list)


@dataclass
class EvaluationReport_v2:
    """Aggregated results across all golden queries for one evaluation run."""

    total_queries: int = 0
    avg_retrieval_recall: float = 0.0
    avg_answer_quality: float = 0.0
    avg_fact_f1: float = 0.0
    avg_latency_ms: float = 0.0
    query_results: list[QueryResult_v2] = field(default_factory=list)
