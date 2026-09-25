"""端到端评估汇总：解析失败的样本必须排除出分母并单独计数。"""
from eval import answer_eval


def _row(ac, t="factoid", f=1.0, p=0.2, r=1.0):
    return {"answer_correctness": ac, "type": t, "faithfulness": f,
            "context_precision": p, "context_recall": r}


def test_summary_excludes_parse_failures_from_denominator():
    s = answer_eval.summarize([_row(1.0), _row(0.0), _row(None)])
    assert s["answer_correctness"] == 0.5          # (1+0)/2，None 不进分母
    assert s["answer_correctness_parse_failures"] == 1
    assert s["n"] == 3


def test_summary_reports_none_when_everything_unparseable():
    s = answer_eval.summarize([_row(None), _row(None)])
    assert s["answer_correctness"] is None
    assert s["answer_correctness_parse_failures"] == 2


def test_summary_per_type_answer_correctness():
    s = answer_eval.summarize([_row(1.0, "factoid"), _row(0.0, "factoid"),
                               _row(1.0, "multi-hop")])
    assert s["per_type"]["factoid"]["answer_correctness"] == 0.5
    assert s["per_type"]["multi-hop"]["answer_correctness"] == 1.0
