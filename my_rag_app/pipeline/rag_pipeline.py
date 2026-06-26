from my_rag_app.retrieval.retriever import HybridRetriever
from my_rag_app.constants import QDRANT_URL, QDRANT_COLLECTION, DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_K_RERANK
from my_rag_app.retrieval.reranker import CrossEncoderReranker
from my_rag_app.prompting.context_builder import ContextBuilder
from my_rag_app.prompting.prompt_builder import PromptBuilder
from my_rag_app.models.load import LLMClient
from my_rag_app.logger import get_logger

# Config
QDRANT_URL      = QDRANT_URL
COLLECTION_NAME = QDRANT_COLLECTION
TOP_K_RETRIEVE  = DEFAULT_TOP_K_RETRIEVE
TOP_K_RERANK    = DEFAULT_TOP_K_RERANK

logger = get_logger(__name__)
# Pipeline wrapper — loads all models/connections once, reused across queries
class EmailAssistant:

    def __init__(self):
        self.retriever       = HybridRetriever(qdrant_url=QDRANT_URL, collection_name=COLLECTION_NAME)
        self.context_builder = ContextBuilder()
        self.prompt_builder  = PromptBuilder()
        self.llm             = LLMClient()
        self.reranker        = CrossEncoderReranker()

    def ask(self, query: str) -> str:
        results     = self.retriever.search(query, top_k=TOP_K_RETRIEVE)
        if not results:
            return "No relevant emails found for this question."

        top_results = self.reranker.rerank(query, results, top_k=TOP_K_RERANK)
        threads     = self.retriever.expand_threads(top_results)
        context     = self.context_builder.build(top_results, threads)
        messages    = self.prompt_builder.build(query, context)
        response    = self.llm.generate(messages)
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
            logger.error("Query failed | error=%s", e)
            print(f"\n[Error: {e}]\n")
            continue

        print(f"\nAssistant: {answer}\n")