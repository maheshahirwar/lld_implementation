from __future__ import annotations

from dataclasses import dataclass

from app.llm_client import BaseLLMClient
from app.vector_store import RetrievedChunk, VectorStore


class ChatServiceError(Exception):
    pass


@dataclass
class ChatService:
    vector_store: VectorStore
    llm_client: BaseLLMClient
    retrieval_k: int
    max_context_chunks: int

    def answer(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.vector_store.search(question, top_k=self.retrieval_k)
        if not retrieved:
            raise ChatServiceError("No indexed PDF content found. Upload a PDF first.")

        context = [chunk.text for chunk in retrieved[: self.max_context_chunks]]
        answer = self.llm_client.answer(question=question, context_chunks=context)
        return answer, retrieved[: self.max_context_chunks]
