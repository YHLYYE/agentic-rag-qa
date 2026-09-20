import faiss

from models import Chunk, RetrievedChunk
from rag.embeddings import Embedder


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, index_path: str):
        self.chunks = chunks
        self.embedder = embedder
        self._index = faiss.read_index(index_path)

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        q = self.embedder.embed([query])[0]
        scores, ids = self._index.search(q, top_k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            out.append(RetrievedChunk(self.chunks[idx], float(score)))
        return out
