from dataclasses import dataclass, field

@dataclass
class IngestionReport:
    inserted: int
    skipped_existing: int
    total_processed: int
    thread_match_method_counts: dict[str, int] = field(default_factory=dict)

@dataclass
class CleaningReport:
    cleaned: int
    system_emails: int
    empty_after_clean: int

@dataclass
class ChunkingReport:
    chunks_created: int
    skipped_empty_body: int

@dataclass
class MetadataReport:
    extracted: int

@dataclass
class QdrantIngestionReport:
    chunks_pending: int
    points_upserted: int
    skipped_empty: int
    batches_failed: int

@dataclass
class PIIMatch:
    kind: str   # "phone" or "email"
    value: str

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""

