from my_rag_app.constants import DEFAULT_TOP_K_RERANK, DEFAULT_TOP_K_RETRIEVE, QDRANT_COLLECTION, QDRANT_URL
from my_rag_app.core.guardrails.pii import PIIDetector
from my_rag_app.core.guardrails.validation import CitationValidator, InputValidator
from my_rag_app.core.prompting.context_builder import ContextBuilder
from my_rag_app.core.prompting.prompt_builder import PromptBuilder
from my_rag_app.core.qdrant.reranker import CrossEncoderReranker
from my_rag_app.core.qdrant.retriever import HybridRetriever
from my_rag_app.logger import get_logger
from my_rag_app.models.load import LLMClient

logger = get_logger(__name__)


# Pipeline wrapper — loads all models/connections once, reused across queries
class EmailAssistant:
    """End-to-end query pipeline: retrieve, rerank, build context, and generate an answer."""

    def __init__(self):
        self.retriever = HybridRetriever(qdrant_url=QDRANT_URL, collection_name=QDRANT_COLLECTION)
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient()
        self.reranker = CrossEncoderReranker()
        self.input_validator = InputValidator()
        self.citation_validator = CitationValidator()
        self.pii_detector = PIIDetector()

    def ask(self, query: str) -> str:
        """Answer a natural-language question against the email knowledge base."""
        input_result = self.input_validator.validate(query)
        if not input_result.is_valid:
            return input_result.reason

        results = self.retriever.search(query, top_k=DEFAULT_TOP_K_RETRIEVE)
        if not results:
            return "No relevant emails found for this question."

        top_results = self.reranker.rerank(query, results, top_k=DEFAULT_TOP_K_RERANK)
        threads = self.retriever.expand_threads(top_results)
        context = self.context_builder.build(top_results, threads)
        messages = self.prompt_builder.build(query, context)
        response = self.llm.generate(messages)

        self.pii_detector.check(response.content)

        citation_result = self.citation_validator.validate(response.content, num_context_emails=len(top_results))
        if not citation_result.is_valid:
            return self.citation_validator.fallback_message()

        return response.content


# Interactive loop
if __name__ == "__main__":
    print("Email Intelligence Assistant — type your question, or 'exit' to quit.\n")

    assistant = EmailAssistant()  # models load once here, not per-query

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        try:
            answer = assistant.ask(query)
        except Exception as e:
            logger.exception("Query failed | error=%s")
            print(f"\n[Error: {e}]\n")
            continue

        print(f"\nAssistant: {answer}\n")
