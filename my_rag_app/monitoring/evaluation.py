"""Batch evaluation pipeline for the RAG system with MLflow tracking on DagsHub."""

import json
import os
import time

import dagshub
import mlflow
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from my_rag_app.constants import (
    CONTEXT_MAX_TOKENS,
    DAGSHUB_TRACKING_URI,
    DEFAULT_TOP_K_RERANK,
    DEFAULT_TOP_K_RETRIEVE,
    DENSE_EMBEDDING_MODEL,
    EXPERIMENT_NAME,
    GOLDEN_PATH,
    LLM_BASE_URL,
    LLM_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    REPORT_PATH,
    RERANKER_MODEL,
    SPARSE_EMBEDDING_MODEL,
)
from my_rag_app.core.prompting.context_builder import ContextBuilder
from my_rag_app.core.prompting.prompt_builder import PromptBuilder
from my_rag_app.core.qdrant.reranker import CrossEncoderReranker
from my_rag_app.core.qdrant.retriever import HybridRetriever
from my_rag_app.entity.reports import EvaluationReport, GoldenQuery, LLMResponse, QueryResult
from my_rag_app.exception.monitoring import GoldenDatasetNotFoundError
from my_rag_app.logger import get_logger
from my_rag_app.models.load import LLMClient

logger = get_logger(__name__)

dagshub.init(repo_owner='danishshaikh06', repo_name='PRODUCTION-READY-RAG', mlflow=True)

# Evaluation pipeline
class EvaluationPipeline:
    """Runs batch evaluation against golden queries and tracks results in MLflow."""

    def __init__(self) -> None:
        """Initialise pipeline components — loaded once, reused across queries."""
        self.retriever = HybridRetriever(
            qdrant_url=os.getenv(QDRANT_URL),
            collection_name=QDRANT_COLLECTION,
        )
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient()
        self.embedding_model = SentenceTransformer(DENSE_EMBEDDING_MODEL)  # BGE model for semantic scoring

    # Entry point
    def run(self) -> EvaluationReport:
        """Load golden queries, evaluate each, log to MLflow, write report."""
        load_dotenv()
        self._configure_mlflow()

        golden_queries = self._load_golden_queries()
        logger.info("Evaluation | %d golden queries loaded", len(golden_queries))

        with mlflow.start_run():
            self._log_params()

            report = EvaluationReport(total_queries=len(golden_queries))

            for idx,gq in golden_queries:
                result = self._evaluate_query(gq)
                report.query_results.append(result)
                self._log_query_metrics(idx,result)

            report = self._compute_aggregates(report)
            self._log_aggregate_metrics(report)

        self._write_report(report)
        logger.info(
            "Evaluation complete | recall=%.3f quality=%.3f latency=%.0fms",
            report.avg_retrieval_recall,
            report.avg_answer_quality,
            report.avg_latency_ms,
        )
        return report

    # MLflow setup
    def _configure_mlflow(self) -> None:
        """Set the DagsHub tracking URI and experiment name."""
        mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    def _log_params(self) -> None:
        """Log pipeline configuration as MLflow params."""
        mlflow.log_params(
            {
                "dense_embedding_model": DENSE_EMBEDDING_MODEL,
                "sparse_embedding_model": SPARSE_EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "llm_model": LLM_MODEL,
                "llm_base_url": LLM_BASE_URL,
                "qdrant_collection": QDRANT_COLLECTION,
                "top_k_retrieve": DEFAULT_TOP_K_RETRIEVE,
                "top_k_rerank": DEFAULT_TOP_K_RERANK,
                "context_max_tokens": CONTEXT_MAX_TOKENS,
                "db_name": os.getenv("DB_NAME", ""),
                "golden_queries": GOLDEN_PATH.name,
            }
        )

    def _log_query_metrics(self, step: int, result: QueryResult) -> None:
        """Log per-query metrics with the query index as a step."""
        mlflow.log_metrics(
            {
                "retrieval_recall": result.retrieval_recall,
                "answer_quality": result.answer_quality,
                "latency_ms": result.latency_ms,
                "query" : result.query,
                "answer": result.answer,
                'retrieved_email_ids': result.retrieved_email_ids,
                'notes': result.notes


            },
            step=step,
        )

    def _log_aggregate_metrics(self, report: EvaluationReport) -> None:
        """Log aggregated metrics for the full evaluation run."""
        mlflow.log_metrics(
            {
                "avg_retrieval_recall": report.avg_retrieval_recall,
                "avg_answer_quality": report.avg_answer_quality,
                "avg_latency_ms": report.avg_latency_ms,
            }
        )
        mlflow.log_artifact(str(REPORT_PATH))

    # Loading
    def _load_golden_queries(self) -> list[GoldenQuery]:
        """Load and parse golden queries from the JSONL file."""
        if not GOLDEN_PATH.exists():
            raise GoldenDatasetNotFoundError(GOLDEN_PATH)

        queries: list[GoldenQuery] = []
        with open(GOLDEN_PATH, encoding="utf-8") as f:
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    queries.append(
                        GoldenQuery(
                            query=data["query"],
                            expected_email_ids=data["expected_email_ids"],
                            expected_answer_contains=data["expected_answer_contains"],
                            notes=data.get("notes", ""),
                        )
                    )
            except Exception:
                logger.exception("Failed to load golden queries from %s", GOLDEN_PATH)
            else:
                return queries

    # Per-query evaluation
    def _evaluate_query(self, gq: GoldenQuery) -> QueryResult:
        """Run the full RAG pipeline for one golden query and score the result."""
        logger.info("Evaluating | query=%r", gq.query[:60])
        start = time.monotonic()

        results = self.retriever.search(gq.query, top_k=DEFAULT_TOP_K_RETRIEVE)
        top_results = self.reranker.rerank(gq.query, results, top_k=DEFAULT_TOP_K_RERANK)
        threads = self.retriever.expand_threads(top_results)
        context = self.context_builder.build(top_results, threads)
        messages = self.prompt_builder.build(gq.query, context)
        response: LLMResponse = self.llm.generate(messages)

        latency_ms = (time.monotonic() - start) * 1000

        retrieved_ids = [r["payload"].get("email_id", "") for r in top_results]
        retrieval_recall = self._compute_recall(gq.expected_email_ids, retrieved_ids)
        answer_quality = self._compute_answer_quality(
            gq.expected_answer_contains, response.content
        )

        logger.info(
            "Query scored | recall=%.2f quality=%.2f latency=%.0fms",
            retrieval_recall,
            answer_quality,
            latency_ms,
        )

        return QueryResult(
            query=gq.query,
            retrieved_email_ids=retrieved_ids,
            answer=response.content,
            retrieval_recall=retrieval_recall,
            answer_quality=answer_quality,
            latency_ms=latency_ms,
            notes=gq.notes,
        )

    # Scoring
    def _compute_recall(
        self, expected_ids: list[str], retrieved_ids: list[str]
    ) -> float:
        """Compute recall: fraction of expected emails found in retrieved results."""
        if not expected_ids:
            return 1.0
        retrieved_set = set(retrieved_ids)
        hits = sum(1 for eid in expected_ids if eid in retrieved_set)
        return hits / len(expected_ids)

    def _compute_answer_quality(
        self, expected_phrases: list[str], answer: str
    ) -> float:
        """Semantic answer quality using BGE embeddings.
           Measures meaning similarity instead of exact string match."""

        if not expected_phrases:
            return 1.0
        if not answer or not answer.strip():
            return 0.0

        answer_text = "passage: " + answer

        phrase_texts = ["query: " + p for p in expected_phrases]

        phrase_embs = self.embedding_model.encode(
        phrase_texts,
        normalize_embeddings=True
        )

        answer_emb = self.embedding_model.encode(
        answer_text,
        normalize_embeddings=True
        )

        scores = cosine_similarity(
        phrase_embs,
        [answer_emb]
        ).flatten()

        return float(scores.mean())

    def _compute_aggregates(self, report: EvaluationReport) -> EvaluationReport:
        """Compute mean metrics across all query results."""
        if not report.query_results:
            return report
        n = len(report.query_results)
        report.avg_retrieval_recall = sum(
            r.retrieval_recall for r in report.query_results
        ) / n
        report.avg_answer_quality = sum(
            r.answer_quality for r in report.query_results
        ) / n
        report.avg_latency_ms = sum(r.latency_ms for r in report.query_results) / n
        return report

    # Report
    def _write_report(self, report: EvaluationReport) -> None:
        """Write the evaluation report as a JSON artifact."""
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_queries": report.total_queries,
            "avg_retrieval_recall": report.avg_retrieval_recall,
            "avg_answer_quality": report.avg_answer_quality,
            "avg_latency_ms": report.avg_latency_ms,
            "queries": [
                {
                    "query": r.query,
                    "retrieval_recall": r.retrieval_recall,
                    "answer_quality": r.answer_quality,
                    "latency_ms": r.latency_ms,
                    "answer": r.answer,
                    "retrieved_email_ids": r.retrieved_email_ids,
                    "notes": r.notes,
                }
                for r in report.query_results
            ],
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Evaluation report written | path=%s", REPORT_PATH)


if __name__ == "__main__":
    EvaluationPipeline().run()
