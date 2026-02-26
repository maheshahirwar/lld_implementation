from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

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
