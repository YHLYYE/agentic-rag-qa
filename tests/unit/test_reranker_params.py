"""重排器的 batch_size / 空输入处理：参数必须真的透传（显存与正确性都靠它）。"""
from models import Chunk, RetrievedChunk
from rag.reranker import Reranker


def _reranker_with_fake_model(scores):
    r = Reranker.__new__(Reranker)          # 跳过加载真实模型
    calls = {}

    class _FakeCE:
        def predict(self, pairs, batch_size=32):
            calls["batch_size"] = batch_size
            calls["n_pairs"] = len(pairs)
            return scores[:len(pairs)]

    r._model = _FakeCE()
    return r, calls


def _cands(n):
    return [RetrievedChunk(Chunk(f"c{i}", f"t{i}", "d", "s", 1), 0.0) for i in range(n)]


def test_batch_size_is_passed_through():
    r, calls = _reranker_with_fake_model([0.5, 0.9])
    r.rerank("q", _cands(2), top_k=2, batch_size=3)
    assert calls["batch_size"] == 3
    assert calls["n_pairs"] == 2


def test_rerank_sorts_by_score_desc():
    r, _ = _reranker_with_fake_model([0.1, 0.9, 0.5])
    out = r.rerank("q", _cands(3), top_k=3)
    assert [rc.chunk.chunk_id for rc in out] == ["c1", "c2", "c0"]


def test_empty_candidates_do_not_call_the_model():
    r, calls = _reranker_with_fake_model([])
    assert r.rerank("q", [], top_k=5) == []
    assert calls == {}          # 空输入不该触发一次模型前向
