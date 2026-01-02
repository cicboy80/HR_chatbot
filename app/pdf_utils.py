import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages in a PDF."""
    if not pdf_path:
        raise ValueError("No PDF file path provided")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(pdf_path)

    text_chunks = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:  # ✅ Prevent NoneType crashes
            text_chunks.append(page_text)

    if not text_chunks:
        raise ValueError("PDF contains no extractable text (possibly scanned image)")

    return "\n".join(text_chunks)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks for embedding."""
    if not text or len(text.strip()) == 0:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks