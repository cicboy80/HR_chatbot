import pytest

from app.pdf_utils import (
    chunk_text,
    clean_extracted_text,
    extract_text_from_pdf,
    split_into_sentences,
)


class TestCleanExtractedText:
    def test_empty_input(self):
        assert clean_extracted_text("") == ""
        assert clean_extracted_text(None) == ""

    def test_joins_hyphenated_line_breaks(self):
        assert "holiday" in clean_extracted_text("holi-\nday entitlement")

    def test_collapses_spaces_and_blank_lines(self):
        cleaned = clean_extracted_text("a   b\t c\n\n\n\nd")
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned


class TestSplitIntoSentences:
    def test_empty(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_basic_split(self):
        sentences = split_into_sentences("First sentence. Second one! Third?")
        assert sentences[0] == "First sentence."
        assert len(sentences) >= 2


class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, overlap=100)

    def test_short_text_single_chunk(self):
        chunks = chunk_text("A short paragraph.", chunk_size=1000, overlap=200)
        assert chunks == ["A short paragraph."]

    def test_long_text_produces_bounded_overlapping_chunks(self):
        paragraphs = "\n\n".join(
            f"Paragraph {i}. " + "This sentence fills space. " * 8
            for i in range(20)
        )
        chunks = chunk_text(paragraphs, chunk_size=500, overlap=100)
        assert len(chunks) > 1
        # every chunk respects the size budget (plus the joined overlap tail)
        assert all(len(c) <= 500 + 100 + 2 for c in chunks)
        # overlap: the tail of chunk N reappears at the start of chunk N+1
        tail = chunks[0][-50:]
        assert tail.strip()[:20] in chunks[1]


class TestExtractTextFromPdf:
    def test_missing_path(self):
        with pytest.raises(ValueError):
            extract_text_from_pdf("")

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/file.pdf")

    def test_corrupt_pdf_raises_value_error(self, tmp_path):
        bad = tmp_path / "corrupt.pdf"
        bad.write_bytes(b"%PDF-1.4 this is not really a pdf at all")
        with pytest.raises(ValueError):
            extract_text_from_pdf(str(bad))
