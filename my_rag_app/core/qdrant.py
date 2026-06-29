"""
Qdrant ingestion — joins chunks with their email + metadata, embeds (dense +
sparse) via fastembed, and upserts into the email_knowledge_v1 collection.
Deterministic point IDs (derived from chunk_id) make re-runs idempotent.
Only chunks with embedded_at IS NULL are processed (incremental).
"""

import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

from my_rag_app.logger import get_logger
from my_rag_app.exception import MyException
from my_rag_app.entity.models import Chunk, Email, Metadata
from my_rag_app.entity.reports import QdrantIngestionReport
from my_rag_app.config.config import get_session
from my_rag_app.constants import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_BATCH_SIZE,DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, DENSE_DIM

logger = get_logger(__name__)

# Deterministic-ID namespace — fixed, arbitrary UUID so point IDs are stable
# across runs (same chunk_id always maps to the same Qdrant point id).
POINT_ID_NAMESPACE = uuid.UUID("a3f5e9c0-1b2d-4e6f-9a8b-7c6d5e4f3a2b")

KEYWORD_INDEX_FIELDS = [
    "sender_email", "sender_company", "sender_designation", "sender_name",
    "thread_id", "recipient_emails", "recipient_names",
]


class QdrantIngestionPipeline:
    """Embeds and upserts chunks (joined with email + metadata) into Qdrant."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.qdrant_url      = qdrant_url
        self.collection_name = collection_name
        self.batch_size      = batch_size

        self.client       = None
        self.dense_model  = None
        self.sparse_model = None

    # Entry point
    def run(self) -> "QdrantIngestionReport":
        logger.info("Starting Qdrant ingestion | collection=%s url=%s", self.collection_name, self.qdrant_url)

        self._connect()
        self._ensure_collection()
        self._ensure_payload_indexes()
        self._load_embedding_models()

        rows = self._load_pending_rows()
        logger.info("Qdrant ingestion | %d chunk(s) pending", len(rows))

        upserted = skipped_empty = batches_failed = 0
        for batch_start in range(0, len(rows), self.batch_size):
            batch = rows[batch_start: batch_start + self.batch_size]
            batch = [r for r in batch if r[0].text.strip()]
            skipped_empty += len(rows[batch_start: batch_start + self.batch_size]) - len(batch)
            if not batch:
                continue

            success = self._process_batch(batch)
            if success:
                upserted += len(batch)
            else:
                batches_failed += 1

        report = QdrantIngestionReport(
            chunks_pending=len(rows),
            points_upserted=upserted,
            skipped_empty=skipped_empty,
            batches_failed=batches_failed,
        )
        logger.info("Qdrant ingestion complete | %s", report)
        return report


    # Setup
    def _connect(self) -> None:
        try:
            self.client = QdrantClient(url=self.qdrant_url)
            self.client.get_collections()
        except Exception as e:
            logger.error("Could not connect to Qdrant at %s | error=%s", self.qdrant_url, e)
            raise MyException(f"Qdrant unreachable at {self.qdrant_url}") from e

    def _ensure_collection(self) -> None:
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in existing:
                logger.info("Collection '%s' already exists - reusing", self.collection_name)
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            logger.info("Created collection '%s'", self.collection_name)
        except Exception as e:
            logger.error("Failed to ensure collection | error=%s", e)
            raise MyException(f"Could not create/verify collection {self.collection_name}") from e

    def _ensure_payload_indexes(self) -> None:
        for field in KEYWORD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug("Payload index skip/exists | field=%s error=%s", field, e)

    def _load_embedding_models(self) -> None:
        try:
            logger.info("Loading dense model: %s", DENSE_EMBEDDING_MODEL)
            self.dense_model = TextEmbedding(model_name= DENSE_EMBEDDING_MODEL)
            logger.info("Loading sparse model: %s", SPARSE_EMBEDDING_MODEL)
            self.sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)
        except Exception as e:
            logger.error("Failed to load embedding models | error=%s", e)
            raise MyException("Could not load embedding models") from e

    # Loading
    def _load_pending_rows(self) -> list[tuple[Chunk, Email, Metadata]]:
        with get_session() as session:
            rows = (
                session.query(Chunk, Email, Metadata)
                .join(Email, Email.id == Chunk.email_id)
                .join(Metadata, Metadata.email_id == Chunk.email_id)
                .filter(Chunk.embedded_at.is_(None))
                .all()
            )
            # Detach values we need so they survive after the session closes
            return [(c, e, m) for c, e, m in rows]

    # Batch processing
    def _process_batch(self, batch: list[tuple[Chunk, Email, Metadata]]) -> bool:
        texts = [chunk.text for chunk, _, _ in batch]

        try:
            dense_vectors  = list(self.dense_model.embed(texts))
            sparse_vectors = list(self.sparse_model.embed(texts))
        except Exception as e:
            logger.error("Embedding failed for batch | error=%s", e)
            return False

        points = []
        for (chunk, email, meta), dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(POINT_ID_NAMESPACE, chunk.chunk_id)),
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload=self._build_payload(chunk, email, meta),
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            logger.error("Upsert failed for batch | error=%s", e)
            return False

        self._mark_embedded([chunk.chunk_id for chunk, _, _ in batch])
        return True

    def _build_payload(self, chunk: Chunk, email: Email, meta: Metadata) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "email_id": chunk.email_id,
            "thread_id": chunk.thread_id,
            "text": chunk.text,
            "date": email.date.isoformat() if email.date else "",
            "sender_email": email.sender_email,
            "recipient_emails": email.recipient_emails,
            "sender_name": meta.sender_name,
            "sender_company": meta.sender_company,
            "sender_designation": meta.sender_designation,
            "recipient_names": meta.recipient_names,
            "greeting_name": meta.greeting_name,
        }

    def _mark_embedded(self, chunk_ids: list[str]) -> None:
        try:
            with get_session() as session:
                session.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).update(
                    {"embedded_at": datetime.now(timezone.utc)},
                    synchronize_session=False,
                )
                session.commit()
        except Exception as e:
            logger.error("Failed to mark chunks as embedded | error=%s", e)
            raise MyException("Could not update embedded_at after upsert") from e


if __name__ == "__main__":
    QdrantIngestionPipeline().run()