"""统一重排流水线：混合召回(top-N) → 去重 → CrossEncoder 精排 → top-k。"""
from rag.pipeline import HybridRerankRetriever


class _Chunk:
    def __init__(self, cid, text="t"):
        self.chunk_id = cid
        self.text = text


class _RC:
    def __init__(self, cid, score):
        self.chunk = _Chunk(cid)
        self.score = score


class _Retriever:
    def __init__(self, cid):
        self.cid = cid
        self.calls = []

    def retrieve(self, q, k):
        self.calls.append(k)
        return [_RC(self.cid, 1.0)]


class _Reranker:
    def __init__(self, prefer):
        self.prefer = prefer           # chunk_id 优先级，靠前者分高
        self.seen = None

    def rerank(self, query, retrieved, top_k):
        self.seen = [rc.chunk.chunk_id for rc in retrieved]
        ordered = sorted(retrieved, key=lambda rc: self.prefer.index(rc.chunk.chunk_id))
        return ordered[:top_k]


def test_recall_stage_uses_recall_k_for_both_paths():
    dense, bm25 = _Retriever("d"), _Retriever("b")
    r = HybridRerankRetriever(dense, bm25, _Reranker(["d", "b"]), recall_k=20, rerank_k=5)
    r.retrieve("q", 5)
    assert dense.calls == [20] and bm25.calls == [20]


def test_fused_candidates_are_deduped_before_rerank():
    """dense 和 bm25 命中同一个 chunk 时只能出现一次。"""
    dense, bm25 = _Retriever("same"), _Retriever("same")
    rer = _Reranker(["same"])
    r = HybridRerankRetriever(dense, bm25, rer, recall_k=10, rerank_k=3)
    r.retrieve("q", 3)
    assert rer.seen == ["same"]


def test_reranker_decides_the_final_order():
    dense, bm25 = _Retriever("d"), _Retriever("b")
    r = HybridRerankRetriever(dense, bm25, _Reranker(["b", "d"]), recall_k=10, rerank_k=1)
    out = r.retrieve("q", 1)
    assert [rc.chunk.chunk_id for rc in out] == ["b"]


def test_returns_at_most_rerank_k():
    dense, bm25 = _Retriever("d"), _Retriever("b")
    r = HybridRerankRetriever(dense, bm25, _Reranker(["b", "d"]), recall_k=10, rerank_k=2)
    assert len(r.retrieve("q", 2)) == 2
