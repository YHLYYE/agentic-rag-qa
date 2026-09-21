import hashlib
import re

from models import Chunk


def make_chunk_id(source_doc: str, page: int, section: str, idx: int) -> str:
    raw = f"{source_doc}|{page}|{section}|{idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, source_doc: str, section: str, page: int,
               chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    words = text.split()
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be > overlap")
    chunks: list[Chunk] = []
    idx = 0
    for i in range(0, max(1, len(words) - overlap), step):
        seg = words[i:i + chunk_size]
        if not seg:
            continue
        chunk_id = make_chunk_id(source_doc, page, section, idx)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=" ".join(seg),
            source_doc=source_doc,
            section=section,
            page=page,
        ))
        idx += 1
    return chunks


_HEADING_RE = re.compile(
    r'(ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+|Consolidated\s+(Balance|Statements?|Statement)|Notes\s+to\s+Consolidated)',
    re.IGNORECASE,
)


def detect_heading(text: str) -> str | None:
    """Return the first 10-K section heading found in a page, or None."""
    for line in text.split("\n"):
        s = line.strip()
        if not s or len(s) > 120:
            continue
        m = _HEADING_RE.search(s)
        if m:
            return m.group(0).strip()
    return None
