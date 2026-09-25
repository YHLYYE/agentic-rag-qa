"""抽样必须分题型均衡 —— 否则小规模评估会得出「单题型 100%」这种假证据。"""
from eval import sampling


def _items(n_per_type: int = 5):
    return ([{"type": "comparison", "i": i} for i in range(n_per_type)]
            + [{"type": "multi-hop", "i": i} for i in range(n_per_type)]
            + [{"type": "factoid", "i": i} for i in range(n_per_type)])


def test_stratified_sample_covers_every_type():
    out = sampling.stratified_sample(_items(5), 2)
    assert len(out) == 6
    assert {o["type"] for o in out} == {"comparison", "factoid", "multi-hop"}


def test_stratified_sample_keeps_order_within_type():
    items = [{"type": "a", "i": i} for i in range(4)]
    assert [o["i"] for o in sampling.stratified_sample(items, 2)] == [0, 1]


def test_stratified_sample_tolerates_type_with_fewer_items():
    items = [{"type": "a", "i": 1}, {"type": "b", "i": 2}, {"type": "b", "i": 3}]
    assert len(sampling.stratified_sample(items, 5)) == 3


def test_stratified_sample_avoids_single_type_prefix():
    """真实场景：eval_set 是 100 comparison + 100 multi-hop + 100 factoid 顺序拼的。"""
    items = ([{"type": "comparison", "i": i} for i in range(100)]
             + [{"type": "multi-hop", "i": i} for i in range(100)]
             + [{"type": "factoid", "i": i} for i in range(100)])
    assert len({o["type"] for o in items[:30]}) == 1                 # 取前缀 = 只有 1 类
    assert len({o["type"] for o in sampling.stratified_sample(items, 10)}) == 3
