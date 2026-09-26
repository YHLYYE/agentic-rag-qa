"""中文 dense 评估：chunk_id 必须与 qrels 的 pid 对得上。

回归背景：曾经用合成 id（p0/p1/...）建 chunk，而 qrels 用真实 pid，
两边永远匹配不上 —— 结果是**所有指标静默变成 0.0**，不报任何错。
"""
from eval import zh_dense_eval as zd


def test_make_chunks_uses_real_pids_as_chunk_ids():
    pids = ["15663", "302162"]
    texts = ["蜂巢快递柜", "产后恢复"]
    chunks = zd.make_chunks(pids, texts)
    assert [c.chunk_id for c in chunks] == pids
    assert [c.text for c in chunks] == texts
    assert [c.source_doc for c in chunks] == pids      # 溯源也用真实 pid


def test_make_chunks_preserves_order_and_length():
    pids = [str(i) for i in range(5)]
    chunks = zd.make_chunks(pids, [f"t{i}" for i in range(5)])
    assert len(chunks) == 5
    assert [c.chunk_id for c in chunks] == pids        # 顺序不能乱，索引要对齐


def test_chunk_ids_would_not_match_qrels_if_synthetic():
    """把这个 bug 的后果写进测试：合成 id 与 qrels pid 交集为空。"""
    pids = ["15663", "302162"]
    synthetic = [f"p{i}" for i in range(len(pids))]
    assert not (set(synthetic) & set(pids))
    assert set(c.chunk_id for c in zd.make_chunks(pids, ["a", "b"])) & set(pids)
