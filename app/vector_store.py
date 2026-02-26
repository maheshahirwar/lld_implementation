from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    text: str
    score: float


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.documents: list[str] = []
        self.vectors = None

    def add_documents(self, docs: list[str]) -> None:
        self.documents.extend(docs)
        self.vectors = self.vectorizer.fit_transform(self.documents)
        self._persist()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.documents or self.vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()

        best_indices = similarities.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(text=self.documents[i], score=float(similarities[i]))
            for i in best_indices
            if similarities[i] > 0
        ]

    def load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("rb") as f:
            payload = pickle.load(f)

        self.documents = payload["documents"]
        self.vectorizer = payload["vectorizer"]
        self.vectors = payload["vectors"]

    def _persist(self) -> None:
        with self.path.open("wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "vectorizer": self.vectorizer,
                    "vectors": self.vectors,
                },
                f,
            )
