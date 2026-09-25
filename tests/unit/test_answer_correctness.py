"""答案正确率判官：解析不出来必须返回 None，绝不能静默当 0 分。"""
from types import SimpleNamespace

from eval import ragas_self


class _Client:
    """假 OpenAI 客户端，固定返回某段文本。"""

    def __init__(self, reply: str):
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=reply))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_yes_verdicts_score_one():
    for reply in ("是", "是的", "Yes", "YES.", "正确"):
        out = ragas_self.answer_correctness(_Client(reply), "q", "a", "gt")
        assert out["score"] == 1.0, reply


def test_no_verdicts_score_zero():
    for reply in ("否", "不是", "No", "NO.", "错误"):
        out = ragas_self.answer_correctness(_Client(reply), "q", "a", "gt")
        assert out["score"] == 0.0, reply


def test_unparseable_reply_returns_none_not_zero():
    out = ragas_self.answer_correctness(_Client("我觉得差不多吧"), "q", "a", "gt")
    assert out["score"] is None      # 不能把「判不出来」记成「答错了」


def test_raw_reply_is_preserved_for_audit():
    out = ragas_self.answer_correctness(_Client("是"), "q", "a", "gt")
    assert out["raw"] == "是"
