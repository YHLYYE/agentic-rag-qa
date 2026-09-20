from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    section: str
    page: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
