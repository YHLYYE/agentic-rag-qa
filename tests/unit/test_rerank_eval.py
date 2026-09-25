"""两段式重排评估：分离「粗排缓存」和「精排打分」，两个模型永不同进程共存。"""
import pickle

from eval import rerank_eval


class _Chunk:
    def __init__(self, cid, text):
        self.chunk_id = cid
        self.text = text


class _RC:
    def __init__(self, cid, text):
        self.chunk = _Chunk(cid, text)
        self.score = 1.0


def test_cache_and_load_roundtrip(tmp_path):
    path = tmp_path / "cands.pkl"
    eval_set = [{"question": "q1", "type": "factoid", "ground_truth": "alpha"}]
    rerank_eval.cache_candidates(eval_set, lambda q, k: [_RC("c1", "alpha")], 20, str(path))
    cache = rerank_eval.load_candidates(str(path))
    assert list(cache) == ["q1"]
    assert cache["q1"][0].chunk.chunk_id == "c1"


def test_rerank_fn_reranks_only_cached_candidates(tmp_path):
    path = tmp_path / "cands.pkl"
    eval_set = [{"question": "q1", "type": "factoid", "ground_truth": "alpha"}]
    rerank_eval.cache_candidates(
        eval_set,
        lambda q, k: [_RC("noise", "nothing"), _RC("gold", "alpha")],
        20, str(path))

    class _Reranker:
        def rerank(self, query, retrieved, top_k):
            ordered = sorted(retrieved, key=lambda rc: rc.chunk.chunk_id != "gold")
            return ordered[:top_k]

    cache = rerank_eval.load_candidates(str(path))
    fn = rerank_eval.make_rerank_fn(cache, _Reranker())
    out = fn("q1", 1)
    assert [rc.chunk.chunk_id for rc in out] == ["gold"]


def test_stage_cache_requests_recall_k(tmp_path):
    path = tmp_path / "cands.pkl"
    seen = []

    def retrieve(q, k):
        seen.append(k)
        return [_RC("c1", "alpha")]

    eval_set = [{"question": "q1", "type": "factoid", "ground_truth": "alpha"}]
    rerank_eval.cache_candidates(eval_set, retrieve, 20, str(path))
    assert seen == [20]
    assert pickle.load(open(path, "rb"))["q1"][0].chunk.chunk_id == "c1"
