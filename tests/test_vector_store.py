from pathlib import Path

from app.vector_store import VectorStore


def test_vector_store_add_and_search(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index.pkl")
    store.add_documents(["python is a programming language", "cats are animals"])

    hits = store.search("what is python", top_k=1)
    assert len(hits) == 1
    assert "python" in hits[0].text
