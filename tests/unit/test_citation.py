from models import Chunk
from rag.citation import format_citation, parse_citations


def test_format_citation():
    c = Chunk("abc123", "text", "apple_10k.pdf", "Item 7", 12)
    assert format_citation(c) == "[apple_10k.pdf §Item 7 p.12]"


def test_parse_citations_extracts_markers():
    text = "Revenue grew 20% {{0123456789abcdef}} driven by demand"
    assert parse_citations(text) == ["0123456789abcdef"]
