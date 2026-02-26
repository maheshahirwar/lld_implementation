from app.pdf_processing import chunk_text


def test_chunk_text_creates_multiple_chunks() -> None:
    text = "Sentence one. " * 200
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_rejects_invalid_settings() -> None:
    try:
        chunk_text("hello", chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
