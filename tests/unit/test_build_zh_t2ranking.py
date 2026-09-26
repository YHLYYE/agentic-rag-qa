"""T2Ranking 子集构建：抽样必须确定性，且子集内 qrels 必须完整。"""
from ingest import build_zh_t2ranking as b


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


QRELS = "qid\t0\tpid\trel\n" + "\n".join([
    "q1\t0\tp1\t3", "q1\t0\tp2\t1", "q1\t0\tp9\t0",
    "q2\t0\tp3\t2", "q2\t0\tp8\t0",
    "q3\t0\tp4\t0", "q3\t0\tp5\t0",          # 全是负例 → 不可用
    "q4\t0\tp6\t1", "q4\t0\tp7\t0",
]) + "\n"


def test_load_qrels_and_queries(tmp_path):
    qp = _write(tmp_path, "queries.dev.tsv", "qid\ttext\nq1\t蜂巢取快递\nq2\t产后肚子\n")
    rp = _write(tmp_path, "qrels.dev.tsv", QRELS)
    assert b.load_queries(qp) == {"q1": "蜂巢取快递", "q2": "产后肚子"}
    qrels = b.load_qrels(rp)
    assert qrels["q1"]["p1"] == 3 and qrels["q3"]["p4"] == 0


def test_select_queries_skips_queries_without_positives(tmp_path):
    qrels = b.load_qrels(_write(tmp_path, "qrels.dev.tsv", QRELS))
    picked = b.select_queries(qrels, 10)
    assert picked == ["q1", "q2", "q4"]      # q3 全是负例，被排除


def test_select_queries_is_deterministic_for_same_seed(tmp_path):
    qrels = b.load_qrels(_write(tmp_path, "qrels.dev.tsv", QRELS))
    assert b.select_queries(qrels, 2, seed=7) == b.select_queries(qrels, 2, seed=7)


def test_select_pids_keeps_all_judged_and_disjoint_distractors(tmp_path):
    qrels = b.load_qrels(_write(tmp_path, "qrels.dev.tsv", QRELS))
    judged, extra = b.select_pids(qrels, ["q1"], 99, seed=1)
    assert judged == {"p1", "p2", "p9"}       # q1 的全部判定段落都要在
    assert not (judged & extra)               # 干扰集与判定集不重叠


def test_extract_subset_streams_only_wanted_pids(tmp_path):
    src = _write(tmp_path, "collection.tsv",
                 "p1\t蜂巢快递柜\np2\t产后恢复\np9\t其他内容\n")
    out = str(tmp_path / "out.tsv")
    n = b.extract_subset(src, {"p1", "p9"}, out)
    assert n == 2
    assert "p2" not in open(out, encoding="utf-8").read()
