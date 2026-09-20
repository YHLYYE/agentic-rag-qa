from rank_bm25 import BM25Okapi

from models import Chunk, RetrievedChunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([c.text.split() for c in chunks])

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked]
