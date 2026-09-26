"""确定性检索指标：不调 LLM、可复现，是每轮优化的主信号。"""
from eval import retrieval_metrics as rm


class _Chunk:
    def __init__(self, text, chunk_id="x"):
        self.text = text
        self.chunk_id = chunk_id


class _RC:
    def __init__(self, text, score=1.0):
        self.chunk = _Chunk(text)
        self.score = score


def test_normalize_collapses_case_and_whitespace():
    assert rm.normalize("  La  Jolie\nFille ") == "la jolie fille"


def test_answer_in_text_true_and_false():
    assert rm.answer_in_text("Discovery Zone", "the Discovery   Zone was founded")
    assert not rm.answer_in_text("Discovery Zone", "nothing relevant here")


def test_empty_ground_truth_never_matches():
    assert not rm.answer_in_text("", "anything")


def test_evaluate_retrieval_hit_rate_mrr_and_answer_ratio():
    items = [
        {"question": "q1", "type": "factoid", "ground_truth": "alpha"},
        {"question": "q2", "type": "factoid", "ground_truth": "beta"},
    ]

    def retrieve(q, k):
        if q == "q1":                      # 答案在第 2 位
            return [_RC("noise"), _RC("alpha here"), _RC("more noise")]
        return [_RC("nothing")]            # 完全没检到

    summary = rm.evaluate_retrieval(retrieve, items, k=3)
    assert summary["n"] == 2
    assert summary["hit_rate@k"] == 0.5
    assert summary["mrr"] == 0.25          # (1/2 + 0) / 2
    assert summary["answer_ratio@k"] == (1 / 3 + 0) / 2


def test_evaluate_retrieval_reports_per_type_and_misses():
    items = [
        {"question": "q1", "type": "factoid", "ground_truth": "alpha"},
        {"question": "q2", "type": "comparison", "ground_truth": "beta"},
    ]

    def retrieve(q, k):
        return [_RC("alpha")] if q == "q1" else [_RC("nothing")]

    summary = rm.evaluate_retrieval(retrieve, items, k=2)
    assert summary["per_type"]["factoid"]["hit_rate@k"] == 1.0
    assert summary["per_type"]["comparison"]["hit_rate@k"] == 0.0
    assert summary["miss_count"] == 1


def test_first_hit_rank_distribution():
    items = [
        {"question": "q1", "type": "factoid", "ground_truth": "alpha"},
        {"question": "q2", "type": "factoid", "ground_truth": "beta"},
    ]

    def retrieve(q, k):
        if q == "q1":
            return [_RC("alpha")]
        return [_RC("x"), _RC("beta")]

    summary = rm.evaluate_retrieval(retrieve, items, k=3)
    assert summary["first_hit_rank"] == {"1": 1, "2": 1}


def test_bench_latency_reports_percentiles_and_calls_every_question():
    calls = []

    def retrieve(q, k):
        calls.append(q)
        return [_RC("x")]

    out = rm.bench_latency(retrieve, ["q1", "q2", "q3", "q4"], k=5, warmup=1)
    assert out["n"] == 4
    assert len(calls) == 5          # warmup 1 次 + 正式 4 次
    for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
        assert out[key] >= 0
