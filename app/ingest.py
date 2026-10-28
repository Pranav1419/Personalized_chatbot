import re
from pypdf import PdfReader


def parse_pdf(path: str) -> list[dict]:
    """Return one dict per page: {"page": 1, "text": "..."}"""
    reader = PdfReader(path)
    return [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # re-join words split by hyphen at line end
    text = re.sub(r"\n+", " ", text)         # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)      # collapse extra spaces
    return text.strip()


def chunk(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Sliding window over characters. Overlap keeps sentences from being cut off from context."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def pdf_to_chunks(path: str) -> list[dict]:
    out = []
    for page in parse_pdf(path):
        for i, c in enumerate(chunk(clean(page["text"]))):
            if len(c.strip()) > 30:  # skip near-empty chunks
                out.append({"text": c, "page": page["page"], "chunk_id": f"p{page['page']}_c{i}"})
    return out
