"""评估取样：小规模抽样必须分题型均衡，不能取前缀。

背景：eval_set 是按 comparison(100) + multi-hop(100) + factoid(100) 顺序拼起来的，
所以 `eval_set[:30]` 拿到的是 30 道 comparison —— 单一题型的「准确率 100%」毫无意义。
"""
from collections import defaultdict


def stratified_sample(items: list[dict], per_type: int, key: str = "type") -> list[dict]:
    """每类取 per_type 条，类内保持原顺序；类型按名称排序，保证结果可比。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item[key]].append(item)
    picked: list[dict] = []
    for t in sorted(buckets):
        picked.extend(buckets[t][:per_type])
    return picked
