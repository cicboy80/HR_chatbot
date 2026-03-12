import os
import re
import pdfplumber


def clean_extracted_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""

    # remove hyphen line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages using pdfplumber."""

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = clean_extracted_text(page_text)
                text_chunks.append(cleaned)

    if not text_chunks:
        raise ValueError("PDF contains no extractable text")

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