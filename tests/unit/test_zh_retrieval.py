"""中文检索指标：对着人工相关性标注算 nDCG/Recall/MRR（不再是字符串匹配代理指标）。"""
from eval import zh_retrieval as zr


def test_dcg_and_ndcg_perfect_ranking_is_one():
    rels = {"p1": 3, "p2": 1}
    assert zr.ndcg_at_k(["p1", "p2"], rels, 2) == 1.0


def test_ndcg_penalises_relevant_doc_ranked_lower():
    rels = {"p1": 3, "p2": 0}
    good = zr.ndcg_at_k(["p1", "p2"], rels, 2)
    bad = zr.ndcg_at_k(["p2", "p1"], rels, 2)
    assert good == 1.0 and bad < good


def test_ndcg_is_zero_when_no_relevant_in_topk():
    rels = {"p1": 2}
    assert zr.ndcg_at_k(["x", "y"], rels, 2) == 0.0


def test_recall_at_k_counts_relevant_found():
    rels = {"p1": 1, "p2": 1, "p3": 1, "p4": 0}
    assert abs(zr.recall_at_k(["p1", "p4"], rels, 2) - 1 / 3) < 1e-9
    assert zr.recall_at_k(["p1", "p2", "p3"], rels, 3) == 1.0


def test_mrr_at_k_uses_first_relevant_rank():
    rels = {"p2": 1}
    assert zr.mrr_at_k(["p1", "p2"], rels, 10) == 0.5
    assert zr.mrr_at_k(["p3"], rels, 10) == 0.0


def test_evaluate_rankings_aggregates_over_queries():
    rankings = {"q1": ["p1"], "q2": ["x"]}
    qrels = {"q1": {"p1": 2}, "q2": {"p2": 2}}
    out = zr.evaluate_rankings(rankings, qrels, ks=(1,))
    assert out["n"] == 2
    assert out["ndcg@1"] == 0.5 and out["recall@1"] == 0.5 and out["mrr@1"] == 0.5


def test_graded_relevance_beats_binary_in_ndcg():
    """4 级相关性必须真的用上：把 rel=3 排前面应优于把 rel=1 排前面。"""
    rels = {"hi": 3, "lo": 1}
    assert zr.ndcg_at_k(["hi", "lo"], rels, 2) > zr.ndcg_at_k(["lo", "hi"], rels, 2)
