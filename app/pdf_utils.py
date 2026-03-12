import os
import re
from pypdf import PdfReader


def clean_extracted_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    if not text:
        return ""

    # Join words broken across line breaks with hyphens: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace line breaks/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Add a space between lowercase-uppercase joins: "informtheMiddle" won't be fixed,
    # but "schoolOffice" -> "school Office"
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
        if page_text:
            cleaned = clean_extracted_text(page_text)
            if cleaned:
                text_chunks.append(cleaned)

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