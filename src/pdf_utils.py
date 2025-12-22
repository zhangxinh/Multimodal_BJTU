from typing import List, Tuple

from pypdf import PdfReader

from . import config


def extract_text_by_page(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


def chunk_pages(pages: List[str]) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    chunk_size = config.CHUNK_SIZE
    max_chunks = config.MAX_CHUNKS_PER_DOC
    for page_idx, text in enumerate(pages):
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        for start in range(0, len(cleaned), chunk_size):
            snippet = cleaned[start : start + chunk_size]
            chunks.append((page_idx + 1, snippet))
            if len(chunks) >= max_chunks:
                return chunks
    return chunks
