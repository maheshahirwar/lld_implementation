from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


class PDFProcessingError(Exception):
    pass


def extract_text_from_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # noqa: BLE001
        raise PDFProcessingError(f"Unable to read PDF: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
        else:
            pages.append(f"[Page {index + 1} had no extractable text]")

    full_text = "\n".join(pages).strip()
    if not full_text:
        raise PDFProcessingError("PDF did not contain extractable text")
    return re.sub(r"\s+", " ", full_text).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if end < text_length:
            last_period = chunk.rfind(".")
            if last_period > chunk_size // 2:
                chunk = chunk[: last_period + 1]
                end = start + len(chunk)

        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, start + 1)

    return chunks
