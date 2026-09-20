from eval.metrics import mrr, hit_rate


def test_mrr():
    rankings = [["a", "b", "c"], ["x", "a"]]
    ground_truth = [{"a"}, {"a"}]
    assert mrr(rankings, ground_truth) == (1.0 + 0.5) / 2


def test_hit_rate():
    rankings = [["a", "b"], ["x", "y"]]
    ground_truth = [{"a"}, {"a"}]
    assert hit_rate(rankings, ground_truth, k=2) == 0.5
