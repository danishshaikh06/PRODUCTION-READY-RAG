from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from my_rag_app.logger import get_logger
from my_rag_app.pipeline.rag_pipeline import EmailAssistant

logger = get_logger(__name__)


# Schema
class ChatMessage(BaseModel):
    """A single turn in the conversation history."""

    role: str
    content: str


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    query: str
    chat_history: list[ChatMessage] = []


class SourceEmail(BaseModel):
    """A single source email surfaced by retrieval."""

    subject: str
    sender: str
    date: str
    snippet: str


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str
    sources: list[SourceEmail]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str


# App lifecycle
assistant: EmailAssistant | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the EmailAssistant on startup and release on shutdown."""
    global assistant
    logger.info("Loading EmailAssistant...")
    assistant = EmailAssistant()
    logger.info("EmailAssistant ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Email Intelligence API",
    description="RAG-powered assistant for SMB Freight FZE aviation email operations.",
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Run the RAG pipeline for a natural-language query and return the answer with sources."""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialised")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("POST /ask | query=%r", request.query[:80])

    history = [{"role": m.role, "content": m.content} for m in request.chat_history]
    answer, sources = assistant.ask(request.query, chat_history=history)

    return AskResponse(
        answer=answer,
        sources=[
            SourceEmail(
                subject=s.get("subject", ""),
                sender=s.get("sender_email", ""),
                date=s.get("date", ""),
                snippet=s.get("text", "")[:200],
            )
            for s in sources
        ],
    )
