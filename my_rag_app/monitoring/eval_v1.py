"""Batch evaluation pipeline for the RAG system with MLflow tracking on DagsHub."""

import json
import os
import time

import dagshub
import mlflow
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from my_rag_app.constants import (
    CONTEXT_MAX_TOKENS,
    DAGSHUB_TRACKING_URI,
    DEFAULT_TOP_K_RERANK,
    DEFAULT_TOP_K_RETRIEVE,
    DENSE_EMBEDDING_MODEL,
    EXPERIMENT_NAME_V1,
    GOLDEN_PATH_V1,
    LLM_MODEL_V2,
    QDRANT_COLLECTION,
    QDRANT_URL,
    REPORT_PATH_V1,
    RERANKER_MODEL,
    SPARSE_EMBEDDING_MODEL,
)
from my_rag_app.core.prompting.context_builder import ContextBuilder
from my_rag_app.core.prompting.prompt_builder import PromptBuilder
from my_rag_app.core.qdrant.reranker import CrossEncoderReranker
from my_rag_app.core.qdrant.retriever import HybridRetriever
from my_rag_app.entity.reports import EvaluationReport_v2, GoldenQuery_V2, LLMResponse, QueryResult_v2
from my_rag_app.exception.monitoring import GoldenDatasetNotFoundError
from my_rag_app.logger import get_logger
from my_rag_app.models.loadv2 import QwenClient

logger = get_logger(__name__)

dagshub.init(repo_owner="danishshaikh06", repo_name="PRODUCTION-READY-RAG", mlflow=True)


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
        self.llm = QwenClient()
        self.embedding_model = SentenceTransformer(
            DENSE_EMBEDDING_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
        )  # BGE model for semantic scoring

        self.query_idx = 0  # For MLflow step tracking

    # Entry point
    def run(self) -> EvaluationReport_v2:
        """Load golden queries, evaluate each, log to MLflow, write report."""
        load_dotenv()
        self._configure_mlflow()

        golden_queries = self._load_golden_queries()
        logger.info("Evaluation | %d golden queries loaded", len(golden_queries))

        with mlflow.start_run():
            self._log_params()

            report = EvaluationReport_v2(total_queries=len(golden_queries))

            for gq in golden_queries:
                result = self._evaluate_query(gq)
                report.query_results.append(result)
                self._log_query_metrics(result)

            report = self._compute_aggregates(report)
            self._log_aggregate_metrics(report)

        self._write_report(report)
        logger.info(
            "Evaluation complete | recall=%.3f quality=%.3f fact_f1=%.3f latency=%.0fms",
            report.avg_retrieval_recall,
            report.avg_answer_quality,
            report.avg_fact_f1,
            report.avg_latency_ms,
        )
        return report

    # MLflow setup
    def _configure_mlflow(self) -> None:
        """Set the DagsHub tracking URI and experiment name."""
        mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME_V1)

    def _log_params(self) -> None:
        """Log pipeline configuration as MLflow params."""
        mlflow.log_params({
            "dense_embedding_model": DENSE_EMBEDDING_MODEL,
            "sparse_embedding_model": SPARSE_EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "llm_model": LLM_MODEL_V2,
            "qdrant_collection": QDRANT_COLLECTION,
            "top_k_retrieve": DEFAULT_TOP_K_RETRIEVE,
            "top_k_rerank": DEFAULT_TOP_K_RERANK,
            "context_max_tokens": CONTEXT_MAX_TOKENS,
            "db_name": os.getenv("DB_NAME", ""),
            "golden_queries": GOLDEN_PATH_V1.name,
        })

    def _log_query_metrics(self, result: QueryResult_v2) -> None:
        """Log per-query metrics with the query index as a step."""
        self.query_idx += 1
        idx = self.query_idx
        mlflow.log_metrics(
            {
                "retrieval_recall": result.retrieval_recall,
                "answer_quality": result.answer_quality_semantic,
                "latency_ms": result.latency_ms,
                "fact_f1": result.fact_f1,
            },
            step=idx,
        )

    def _log_aggregate_metrics(self, report: EvaluationReport_v2) -> None:
        """Log aggregated metrics for the full evaluation run."""
        mlflow.log_metrics({
            "avg_retrieval_recall": report.avg_retrieval_recall,
            "avg_answer_quality": report.avg_answer_quality,
            "avg_fact_f1": report.avg_fact_f1,
            "avg_latency_ms": report.avg_latency_ms,
        })
        mlflow.log_artifact(str(REPORT_PATH_V1))

    # Loading
    def _load_golden_queries(self) -> list[GoldenQuery_V2]:
        """Load and parse golden queries from the JSONL file."""
        if not GOLDEN_PATH_V1.exists():
            raise GoldenDatasetNotFoundError(GOLDEN_PATH_V1)

        queries: list[GoldenQuery_V2] = []
        with open(GOLDEN_PATH_V1, encoding="utf-8") as f:
            try:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    queries.append(
                        GoldenQuery_V2(
                            query=data["query"],
                            ground_truth_context=data["ground_truth_context"],
                            ground_truth_answer=data["ground_truth_answer"],
                            key_facts=data["key_facts"],
                            metadata=data["metadata"],
                        )
                    )
            except Exception:
                logger.exception("Failed to load golden queries from %s", GOLDEN_PATH_V1)
            else:
                return queries

    # Per-query evaluation
    def _evaluate_query(self, gq: GoldenQuery_V2) -> QueryResult_v2:
        """Run the full RAG pipeline for one golden query and score the result."""
        logger.info("Evaluating | query=%r", gq.query[:60])
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all GPU operations are complete before measuring latency
        start = time.monotonic()

        results = self.retriever.search(gq.query, top_k=DEFAULT_TOP_K_RETRIEVE)
        top_results = self.reranker.rerank(gq.query, results, top_k=DEFAULT_TOP_K_RERANK)
        threads = self.retriever.expand_threads(top_results)
        context = self.context_builder.build(top_results, threads)
        messages = self.prompt_builder.build(gq.query, context)
        response: LLMResponse = self.llm.generate(messages)

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all GPU operations are complete before measuring latency
        latency_ms = (time.monotonic() - start) * 1000

        retrieved_ids = [r["payload"].get("email_id", "") for r in top_results]
        retrieval_recall = self._compute_recall(gq.ground_truth_context, retrieved_ids)
        answer_quality, fact_f1 = self._compute_answer_quality(gq.ground_truth_answer, gq.key_facts, response.content)

        logger.info(
            "Query scored | recall=%.2f quality=%.2f latency=%.0fms",
            retrieval_recall,
            answer_quality,
            latency_ms,
        )

        return QueryResult_v2(
            query=gq.query,
            retrieved_email_ids=retrieved_ids,
            answer=response.content,
            retrieval_recall=retrieval_recall,
            answer_quality_semantic=answer_quality,
            latency_ms=latency_ms,
            fact_f1=fact_f1,
        )

    # Scoring
    def _compute_recall(self, expected_ids: list[str], retrieved_ids: list[str]) -> float:
        """Compute recall: fraction of expected emails found in retrieved results."""
        if not expected_ids:
            return 1.0
        retrieved_set = set(retrieved_ids)
        hits = sum(1 for eid in expected_ids if eid in retrieved_set)
        return hits / len(expected_ids)

    def _compute_answer_quality(
        self, ground_truth_answers: str, key_facts: list[str], answer: str
    ) -> tuple[float, float]:
        """Semantic answer quality using BGE embeddings.
        Measures meaning similarity instead of exact string match."""

        if not answer or not answer.strip():
            return 0.0, 0.0

        answer_text = "passage: " + answer

        phrase_texts = ["query: " + ground_truth_answers]

        phrase_embs = self.embedding_model.encode(phrase_texts, normalize_embeddings=True)

        answer_emb = self.embedding_model.encode(answer_text, normalize_embeddings=True)

        scores = cosine_similarity(phrase_embs, [answer_emb]).flatten()

        if not key_facts:
            fact_f1 = 1.0
        else:
            found = sum(1 for fact in key_facts if fact.lower() in answer.lower())
            fact_f1 = found / len(key_facts)

        return float(scores.mean()), float(fact_f1)

    def _compute_aggregates(self, report: EvaluationReport_v2) -> EvaluationReport_v2:
        """Compute mean metrics across all query results."""
        if not report.query_results:
            return report
        n = len(report.query_results)
        report.avg_retrieval_recall = sum(r.retrieval_recall for r in report.query_results) / n
        report.avg_answer_quality = sum(r.answer_quality_semantic for r in report.query_results) / n
        report.avg_fact_f1 = sum(r.fact_f1 for r in report.query_results) / n
        report.avg_latency_ms = sum(r.latency_ms for r in report.query_results) / n
        return report

    # Report
    def _write_report(self, report: EvaluationReport_v2) -> None:
        """Write the evaluation report as a JSON artifact."""
        REPORT_PATH_V1.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_queries": report.total_queries,
            "avg_retrieval_recall": report.avg_retrieval_recall,
            "avg_answer_quality": report.avg_answer_quality,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_fact_f1": report.avg_fact_f1,
            "queries": [
                {
                    "query": r.query,
                    "retrieval_recall": r.retrieval_recall,
                    "answer_quality_semantic": r.answer_quality_semantic,
                    "fact_f1": r.fact_f1,
                    "latency_ms": r.latency_ms,
                    "answer": r.answer,
                    "retrieved_email_ids": r.retrieved_email_ids,
                }
                for r in report.query_results
            ],
        }
        with open(REPORT_PATH_V1, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Evaluation report written | path=%s", REPORT_PATH_V1)


if __name__ == "__main__":
    EvaluationPipeline().run()
