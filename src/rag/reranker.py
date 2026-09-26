"""Learned reranker (bge-reranker) for precise chunk ranking."""
from sentence_transformers import CrossEncoder

from models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "data/models/bge-reranker-base",
                 max_length: int | None = None, device: str | None = None):
        # max_length 必须可控：候选序列很长时（本项目中文语料 p99 达 6449 字符），
        # CrossEncoder 的中间激活会直接把显存打爆（实测重排 100 个候选时
        # PyTorch 占到 10.66GB > 8GB 显存）。
        kwargs = {"device": device} if device else {}
        if max_length:
            kwargs["max_length"] = max_length
        self._model = CrossEncoder(model_name, **kwargs)

    def rerank(self, query: str, retrieved: list[RetrievedChunk],
               top_k: int = 5, batch_size: int = 8) -> list[RetrievedChunk]:
        pairs = [(query, rc.chunk.text) for rc in retrieved]
        if not pairs:
            return []
        scores = self._model.predict(pairs, batch_size=batch_size)
        ranked = sorted(zip(scores, retrieved), key=lambda x: -x[0])
        return [rc for _, rc in ranked[:top_k]]
