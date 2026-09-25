"""run_graph 是「人能跑的端到端入口」，测试用桩组件覆盖它的编排与追踪渲染。"""
import pytest

from models import Chunk, RetrievedChunk

import run_graph


def _retrievers():
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    return {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
            "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}


class _ScriptedLLM:
    def __init__(self, semantic_intent: str = "factoid",
                 answer: str = "Revenue grew 20% {{a}}"):
        self.semantic_intent = semantic_intent
        self.answer = answer

    def complete(self, prompt: str) -> str:
        if "只输出类别名" in prompt:
            return self.semantic_intent
        return self.answer


def test_run_returns_final_state_through_the_graph():
    state = run_graph.run("did revenue grow?", _retrievers(), _ScriptedLLM())
    assert state["route_decision"]["source"] == "llm"
    assert state["grounding_verdict"] == "supported"
    assert state["final_answer"] == "Revenue grew 20% {{a}}"


def test_run_uses_hybrid_for_multihop():
    state = run_graph.run("did revenue grow?", _retrievers(),
                          _ScriptedLLM("multi-hop"))
    assert state["intent"] == "hybrid"


def test_run_degrades_instead_of_looping_when_citations_never_ground():
    bad = _ScriptedLLM(answer="Revenue grew 20% {{deadbeef}}")
    state = run_graph.run("did revenue grow?", _retrievers(), bad, max_retry=1)
    assert state["grounding_verdict"] == "unsupported"
    assert state["final_answer"]        # 降级文案，而不是空手而归
    assert state["retry_count"] == 1


def test_run_surfaces_clarification_for_ambiguous_question():
    state = run_graph.run("who?", _retrievers(), None)
    assert "__interrupt__" in state     # 走 clarify 分支暂停，而不是硬着头皮答
    assert not state["final_answer"]


def test_format_trace_covers_all_five_nodes():
    state = run_graph.run("did revenue grow?", _retrievers(), _ScriptedLLM())
    text = "\n".join(run_graph.format_trace(state))
    for marker in ("[1] 路由", "[2] 检索", "[3] 检索批判", "[4] 生成", "[5] 引用校验"):
        assert marker in text
    assert "supported" in text
    assert "Revenue grew 20%" in text


def test_initial_state_has_every_field_the_graph_needs():
    state = run_graph.initial_state("q")
    for field in ("question", "intent", "route_decision", "retrieved_chunks",
                  "retrieval_verdict", "candidate_answer", "citations",
                  "grounding_verdict", "final_answer", "retry_count",
                  "needs_clarify", "clarification"):
        assert field in state


def test_build_parser_defaults_to_no_llm_flag_off():
    args = run_graph.build_parser().parse_args(["did revenue grow?"])
    assert args.question == "did revenue grow?"
    assert args.no_llm is False
    assert args.max_retry == 2


# --- 离线模式：没有 API key 也要能跑完整图（含引用硬闸门）---

def test_extractive_llm_cites_a_real_chunk_from_the_prompt():
    prompt = ("Question: q\n\nContext:\n[abcdef012345] revenue grew 20%\n\n"
              "Answer concisely and cite sources inline as {{chunk_id}}.")
    out = run_graph.ExtractiveLLM().complete(prompt)
    assert "{{abcdef012345}}" in out      # 引用必须来自上下文里真实存在的 chunk_id
    assert "revenue grew 20%" in out


def test_extractive_llm_routes_by_rule_when_asked_to_classify():
    prompt = "把以下问题分类为三类之一，只输出类别名（factoid / comparison / multi-hop）："
    assert run_graph.ExtractiveLLM().complete(prompt) == "factoid"


def test_offline_mode_runs_the_whole_graph_without_an_api_key():
    state = run_graph.run("did revenue grow?", _retrievers(), run_graph.ExtractiveLLM())
    assert state["grounding_verdict"] == "supported"
    assert state["final_answer"]
