import os
import re
import pdfplumber

def clean_extracted_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    if not text:
        return ""

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages using pdfplumber."""
    if not pdf_path:
        raise ValueError("No PDF file path provided")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = clean_extracted_text(page_text)
                if cleaned:
                    text_chunks.append(cleaned)

    if not text_chunks:
        raise ValueError("PDF contains no extractable text (possibly scanned image)")

    return "\n\n".join(text_chunks)


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on punctuation followed by space + capital/open quote/bracket
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"(\[])', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Chunk by paragraphs first, then sentences if needed.
    Produces cleaner semantic chunks than raw character slicing.
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            # Break oversized paragraphs into sentence groups
            sentences = split_into_sentences(para)
            if not sentences:
                units.append(para[:chunk_size])
                continue

            current = ""
            for sent in sentences:
                if not current:
                    current = sent
                elif len(current) + 1 + len(sent) <= chunk_size:
                    current += " " + sent
                else:
                    units.append(current.strip())
                    current = sent
            if current:
                units.append(current.strip())

    # Merge units into chunks with soft overlap
    chunks: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= chunk_size:
            current += "\n\n" + unit
        else:
            chunks.append(current.strip())

            # overlap by trailing characters from previous chunk
            tail = current[-overlap:].strip()
            current = (tail + "\n\n" + unit).strip() if tail else unit

    if current:
        chunks.append(current.strip())

    return chunks