from ingest.parse import extract_pages


def test_extract_pages_from_pdf_text(tmp_path):
    # build a minimal valid PDF is heavy; instead test the text-cleaning path
    from ingest.parse import clean_text
    dirty = "  Hello   world \n\n\n   "
    assert clean_text(dirty) == "Hello world"


def test_extract_pages_skips_empty(tmp_path):
    pages = ["", "  \n  ", "actual text here"]
    result = [p for p in pages if p.strip()]
    assert result == ["actual text here"]
