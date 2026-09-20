import re

from models import Chunk


def format_citation(chunk: Chunk) -> str:
    return f"[{chunk.source_doc} §{chunk.section} p.{chunk.page}]"


def parse_citations(text: str) -> list[str]:
    """Extract chunk_id markers of the form {{chunk_id}} from an answer."""
    return re.findall(r"\{\{([a-f0-9]+)\}\}", text)
