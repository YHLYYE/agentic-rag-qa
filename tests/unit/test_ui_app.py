"""界面已改为复用图链路：用 Streamlit 官方 AppTest 做无浏览器冒烟测试。

重点验证两件事：
1. 脚本能跑起来、不抛异常（API 用错会立刻暴露）
2. 提交问题后，走的是 run_graph 的图链路，并渲染出答案与追踪
"""
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

# AppTest 按「调用文件」解析相对路径，所以这里必须用绝对路径
APP = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")


_FAKE_STATE = {
    "question": "q",
    "route_decision": {"semantic_intent": "factoid", "strategy": "keyword", "source": "llm"},
    "intent": "keyword",
    "retrieved_chunks": [{"chunk_id": "abc123", "text": "the telephone was invented", "score": 1.0}],
    "retrieval_verdict": "correct",
    "candidate_answer": "Bell {{abc123}}",
    "citations": ["abc123"],
    "grounding_verdict": "supported",
    "final_answer": "Bell invented it {{abc123}}",
    "retry_count": 1,
    "needs_clarify": False,
    "clarification": "",
}


def _patched(**overrides):
    """把界面依赖的重活全部换成假实现，避免在测试里加载模型/调 API。"""
    base = {
        "run_graph.load_retrievers": {},
        "run_graph.build_llm": object(),
        "run_graph.build_reranker": None,
        "run_graph.run": _FAKE_STATE,
    }
    base.update(overrides)
    return [patch(k, return_value=v) for k, v in base.items()]


def test_app_renders_without_exception():
    with patch("run_graph.load_retrievers", return_value={}), \
         patch("run_graph.build_llm", return_value=object()):
        at = AppTest.from_file(APP)
        at.run()
    assert not at.exception
    assert len(at.text_input) == 1


def test_submitting_uses_the_graph_and_renders_answer_and_trace():
    patches = _patched()
    for p in patches:
        p.start()
    try:
        at = AppTest.from_file(APP)
        at.run()
        at.text_input[0].set_value("who invented the telephone?")
        at.button[0].click()
        at.run()
    finally:
        for p in patches:
            p.stop()

    assert not at.exception
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Bell invented it" in markdown_text      # 渲染了最终答案
    assert "链路追踪" in markdown_text               # 渲染了图链路追踪
    assert "[5] 引用校验" in "\n".join(t.value for t in at.text)


def test_refusal_state_is_rendered_gracefully():
    refused = dict(_FAKE_STATE, grounding_verdict="unsupported",
                   final_answer="抱歉，知识库里没有检索到能回答这个问题的资料。",
                   citations=[], candidate_answer="")
    patches = _patched(**{"run_graph.run": refused})
    for p in patches:
        p.start()
    try:
        at = AppTest.from_file(APP)
        at.run()
        at.text_input[0].set_value("who invented the telephone?")
        at.button[0].click()
        at.run()
    finally:
        for p in patches:
            p.stop()

    assert not at.exception
    assert "没有检索到" in "\n".join(m.value for m in at.markdown)
