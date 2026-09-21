from models import Chunk, RetrievedChunk
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank


def _chunks():
    return [
        Chunk("a", "revenue grew 20 percent in 2024", "d.pdf", "s1", 1),
        Chunk("b", "operating income was stable", "d.pdf", "s2", 2),
        Chunk("c", "the company sells furniture", "d.pdf", "s3", 3),
    ]


def test_bm25_retriever_returns_relevant():
    r = BM25Retriever(_chunks())
    out = r.retrieve("revenue growth 2024", top_k=2)
    assert out[0].chunk.chunk_id == "a"
    assert all(isinstance(x, RetrievedChunk) for x in out)


def test_merge_and_rerank_dedups():
    cs = _chunks()
    r1 = [RetrievedChunk(cs[0], 0.9), RetrievedChunk(cs[1], 0.5)]
    r2 = [RetrievedChunk(cs[1], 0.8), RetrievedChunk(cs[2], 0.4)]
    merged = merge_and_rerank([r1, r2], top_k=3)
    ids = [m.chunk.chunk_id for m in merged]
    assert len(ids) == len(set(ids))  # dedup
    assert ids[0] == "b"  # b appears in both lists → highest fused (RRF) rank
