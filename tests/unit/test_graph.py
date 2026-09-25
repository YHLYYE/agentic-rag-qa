from graph.build import build_graph


def test_graph_builds_and_has_nodes():
    graph = build_graph(retrievers={"dense": lambda q, k: [], "bm25": lambda q, k: []},
                        llm=None, max_retry=2)
    names = [n for n in graph.nodes]
    for expected in ["route", "retrieve", "critique", "generate", "verify"]:
        assert expected in names


def test_graph_invokes_end_to_end_happy_path():
    from models import Chunk, RetrievedChunk
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    retrievers = {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
                  "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}
    graph = build_graph(retrievers=retrievers, llm=_LLM(), max_retry=2)
    result = graph.invoke({"question": "did revenue grow?",
                           "intent": "hybrid", "retry_count": 0,
                           "retrieved_chunks": [], "citations": [],
                           "retrieval_verdict": "", "grounding_verdict": "",
                           "candidate_answer": "", "final_answer": "",
                           "route_decision": {}})
    assert result["grounding_verdict"] == "supported"


class _LLM:
    def complete(self, prompt: str) -> str:
        return "Yes, revenue grew 20% {{a}}"


# --- 回归：引用永远不命中时，图必须有界终止并给出降级答案 ---

class _BogusCitationLLM:
    """总是引用一个从未被检索到的 chunk，逼 verify 判 unsupported。"""

    def complete(self, prompt: str) -> str:
        return "Revenue grew 20% {{deadbeef}}"


def _stub_retrievers(chunks):
    from models import RetrievedChunk
    from rag.bm25 import BM25Retriever
    r = BM25Retriever(chunks)
    return {"dense": r.retrieve, "bm25": r.retrieve}


def _initial_state(intent: str = "semantic") -> dict:
    return {"question": "did revenue grow?", "intent": intent, "retry_count": 0,
            "retrieved_chunks": [], "citations": [], "retrieval_verdict": "",
            "grounding_verdict": "", "candidate_answer": "", "final_answer": "",
            "route_decision": {}, "needs_clarify": False, "clarification": ""}


def test_graph_bounded_when_citations_never_ground():
    """修复前：retry_count 从不自增 → 回退环无界，实测 2501 次 LLM 调用后 GraphRecursionError。"""
    from models import Chunk
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    g = build_graph(_stub_retrievers(chunks), _BogusCitationLLM(), max_retry=2)
    out = g.invoke(_initial_state())
    assert out["grounding_verdict"] == "unsupported"
    assert out["retry_count"] <= 2
    assert out.get("final_answer")  # 超限要降级，不能空手而归


def test_graph_honors_max_retry_param():
    from models import Chunk
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    g = build_graph(_stub_retrievers(chunks), _BogusCitationLLM(), max_retry=1)
    out = g.invoke(_initial_state())
    assert out["retry_count"] == 1
    assert out.get("final_answer")


# --- 回归：图必须真的用上 LLM 路由（否则 84.3% 的路由能力在演示里是死代码）---

class _RouterAndAnswerLLM:
    """路由 prompt → 返回指定类别；生成 prompt → 返回带真实引用的答案。"""

    def __init__(self, semantic_intent: str = "multi-hop"):
        self.semantic_intent = semantic_intent

    def complete(self, prompt: str) -> str:
        if "只输出类别名" in prompt:
            return self.semantic_intent
        return "Revenue grew 20% {{a}}"


def test_graph_uses_llm_router_when_llm_provided():
    from models import Chunk, RetrievedChunk
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    seen = []

    def recorder(name):
        def _retrieve(q, k):
            seen.append(name)
            return [RetrievedChunk(chunks[0], 0.9)]
        return _retrieve

    g = build_graph({"dense": recorder("dense"), "bm25": recorder("bm25")},
                    _RouterAndAnswerLLM("multi-hop"), max_retry=2)
    out = g.invoke(_initial_state())

    assert out["route_decision"]["source"] == "llm"      # 走的是 LLM 分类，不是规则兜底
    assert out["intent"] == "hybrid"                     # multi-hop → 混合检索
    assert seen == ["dense", "bm25"]                     # 且真的双路都调了
    assert out["grounding_verdict"] == "supported"
