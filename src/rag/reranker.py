"""Learned reranker (bge-reranker) for precise chunk ranking."""
from sentence_transformers import CrossEncoder

from models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "data/models/bge-reranker-base"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved: list[RetrievedChunk],
               top_k: int = 5) -> list[RetrievedChunk]:
        pairs = [(query, rc.chunk.text) for rc in retrieved]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, retrieved), key=lambda x: -x[0])
        return [rc for _, rc in ranked[:top_k]]
