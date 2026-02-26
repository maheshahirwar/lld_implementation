from __future__ import annotations

from dataclasses import dataclass

import httpx
from openai import OpenAI


class LLMConfigurationError(Exception):
    pass


class BaseLLMClient:
    def answer(self, question: str, context_chunks: list[str]) -> str:
        raise NotImplementedError


@dataclass
class OpenAILLMClient(BaseLLMClient):
    api_key: str
    model: str

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def answer(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(f"Source {idx + 1}: {chunk}" for idx, chunk in enumerate(context_chunks))

        system_prompt = (
            "You are a helpful assistant that answers strictly based on provided PDF context. "
            "If the answer is not in the context, respond: 'I don't have enough information from the PDF to answer this.'"
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer concisely and cite which source numbers were used."
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or "No answer generated."


@dataclass
class OllamaLLMClient(BaseLLMClient):
    base_url: str
    model: str

    def answer(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(f"Source {idx + 1}: {chunk}" for idx, chunk in enumerate(context_chunks))

        system_prompt = (
            "You are a helpful assistant that answers strictly based on provided PDF context. "
            "If the answer is not in the context, respond: 'I don't have enough information from the PDF to answer this.'"
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer concisely and cite which source numbers were used."
        )

        response = httpx.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()

        message = payload.get("message", {})
        content = message.get("content", "")
        return content or "No answer generated."


def build_llm_client(settings: object) -> BaseLLMClient:
    provider = getattr(settings, "llm_provider", "openai").strip().lower()

    if provider == "openai":
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not configured")
        return OpenAILLMClient(api_key=api_key, model=getattr(settings, "openai_model"))

    if provider == "ollama":
        return OllamaLLMClient(
            base_url=getattr(settings, "ollama_base_url"),
            model=getattr(settings, "ollama_model"),
        )

    raise LLMConfigurationError("Unsupported LLM_PROVIDER. Use 'openai' or 'ollama'.")
