"""统一重排流水线：混合召回（dense + BM25，RRF 融合）→ 去重 → CrossEncoder 精排。

为什么要拆成「粗排 → 精排」两段：
- 向量/BM25 是**粗排**：快、能上大规模，但排序精度有限（本项目实测：15.3% 的题答案
  落在 top-5 之外，其中 69.6% 其实在 top-50 里 —— 典型的排序问题，不是召回不到）。
- CrossEncoder 是**精排**：把 query 和 chunk 拼在一起编码，精度高得多，但算不了大规模，
  所以只在粗排后的候选集（默认 20）上跑。

这样做的代价是延迟：每题多一次 reranker 前向（20 个 pair）。
"""
from __future__ import annotations

from models import RetrievedChunk
from rag.hybrid import merge_and_rerank


class HybridRerankRetriever:
    """粗排（hybrid 召回到 recall_k）→ 精排（rerank 到 k）。"""

    def __init__(self, dense, bm25, reranker, recall_k: int = 20, rerank_k: int = 5,
                 rrf_k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.recall_k = recall_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.rerank_k
        fused = merge_and_rerank(
            [self.dense.retrieve(query, self.recall_k),
             self.bm25.retrieve(query, self.recall_k)],
            top_k=self.recall_k, k=self.rrf_k,
        )
        return self.reranker.rerank(query, fused, top_k=k)
