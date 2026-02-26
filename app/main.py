from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.chat_service import ChatService, ChatServiceError
from app.config import get_settings
from app.llm_client import LLMClient
from app.pdf_processing import PDFProcessingError, chunk_text, extract_text_from_pdf
from app.schemas import ChatRequest, ChatResponse, IngestResponse, SourceChunk
from app.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("pdf_qa_bot")

settings = get_settings()
vector_store = VectorStore(settings.index_path)
vector_store.load()

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text)
        vector_store.add_documents(chunks)
    except (PDFProcessingError, ValueError) as exc:
        logger.exception("Failed to process PDF")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info("Indexed %s chunks for file %s", len(chunks), file.filename)
    return IngestResponse(
        filename=file.filename or "uploaded.pdf",
        chunks_indexed=len(chunks),
        message="PDF indexed successfully",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    chat_service = ChatService(
        vector_store=vector_store,
        llm_client=LLMClient(api_key=settings.openai_api_key, model=settings.openai_model),
        retrieval_k=settings.retrieval_k,
        max_context_chunks=settings.max_context_chunks,
    )

    try:
        answer, sources = chat_service.answer(payload.question)
    except ChatServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        sources=[SourceChunk(text=s.text, score=s.score) for s in sources],
    )
