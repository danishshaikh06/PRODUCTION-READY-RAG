import torch
from sentence_transformers import CrossEncoder as STCrossEncoder

from my_rag_app.constants import DEFAULT_TOP_K_RERANK, RERANKER_MODEL
from my_rag_app.logger import get_logger

logger = get_logger(__name__)

# Config
MODEL_NAME = RERANKER_MODEL

_model = None  # lazy-loaded module-level singleton — avoids reloading on every call


class CrossEncoderReranker:
    """Re-scores search results against the query using a cross-encoder model."""

    def __init__(self):
        self._get_model()
        print(f"Cross-encoder model loaded on device: {next(_model.model.parameters()).device}")

    def _get_model(self) -> STCrossEncoder:
        global _model
        if _model is None:
            logger.info("Loading cross-encoder model: %s", MODEL_NAME)
            with torch.inference_mode():
                _model = STCrossEncoder(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", num_labels=1)
                _model.model.eval()
                _model.model.half()
        return _model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = DEFAULT_TOP_K_RERANK,
    ) -> list[dict]:
        """Return the top_k results re-ranked by cross-encoder relevance score."""
        if not results:
            return []

        if not query.strip():
            logger.warning("rerank called with empty query")
            return results[:top_k]

        try:
            model = self._get_model()
            pairs = [(query, r["payload"].get("text", "")) for r in results]
            scores = model.predict(
                pairs,
                batch_size=16,
                convert_to_numpy=True,
                show_progress_bar=False,)
        except Exception:
            logger.exception("Reranking failed | %s")
            return results[:top_k]

        reranked = [{"score": float(score), "payload": r["payload"]} for r, score in zip(results, scores)]

        reranked.sort(key=lambda r: r["score"], reverse=True)
        return reranked[:top_k]
