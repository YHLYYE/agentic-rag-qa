from graph.nodes import route_node, verify_node


def test_route_node_semantic():
    state = {"question": "How does revenue relate to profit?"}
    out = route_node(state)
    assert out["intent"] == "semantic"


def test_route_node_keyword_rule_fallback():
    state = {"question": "What is the definition of gross margin?"}
    out = route_node(state)
    assert out["intent"] == "keyword"


def test_verify_node_supported():
    state = {
        "candidate_answer": "revenue grew {{abc123}}",
        "citations": ["abc123"],
        "retrieved_chunks": [{"chunk_id": "abc123"}],
    }
    out = verify_node(state)
    assert out["grounding_verdict"] == "supported"


def test_verify_node_unsupported():
    state = {
        "candidate_answer": "revenue grew {{deadbeef}}",
        "citations": ["deadbeef"],
        "retrieved_chunks": [{"chunk_id": "abc123"}],
    }
    out = verify_node(state)
    assert out["grounding_verdict"] == "unsupported"


from models import Chunk, RetrievedChunk
from graph.nodes import (retrieve_node, generate_node, critique_node, give_up_node,
                         rewrite_node)


class _FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Yes, revenue grew by 20% {{a}}"


def test_retrieve_node_uses_retrievers():
    chunks = [Chunk("a", "revenue grew", "d", "s", 1)]
    retrievers = {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
                  "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}
    state = {"question": "revenue", "intent": "hybrid", "retrieved_chunks": []}
    out = retrieve_node(state, retrievers=retrievers)
    assert len(out["retrieved_chunks"]) == 1


def test_critique_node_returns_verdict():
    state = {"question": "did revenue grow?",
             "retrieved_chunks": [{"chunk_id": "a", "text": "revenue grew 20%"}]}
    out = critique_node(state)
    assert out["retrieval_verdict"] in {"correct", "ambiguous", "incorrect"}


def test_generate_node_produces_citations():
    state = {"question": "did revenue grow?",
             "retrieved_chunks": [{"chunk_id": "a", "text": "revenue grew 20%"}]}
    out = generate_node(state, llm=_FakeLLM())
    assert "{{a}}" in out["candidate_answer"]
    assert out["citations"] == ["a"]


class _FakeRouteLLM:
    def __init__(self, intent: str):
        self.intent = intent

    def complete(self, prompt: str) -> str:
        return self.intent


def test_route_node_llm_factoid_maps_to_keyword():
    out = route_node({"question": "who invented the telephone"}, llm=_FakeRouteLLM("factoid"))
    assert out["intent"] == "keyword"
    assert out["route_decision"]["semantic_intent"] == "factoid"
    assert out["needs_clarify"] is False


def test_route_node_llm_comparison_maps_to_semantic():
    out = route_node({"question": "compare A and B"}, llm=_FakeRouteLLM("comparison"))
    assert out["intent"] == "semantic"


def test_route_node_llm_multihop_maps_to_hybrid():
    out = route_node({"question": "who directed the movie A starred in"}, llm=_FakeRouteLLM("multi-hop"))
    assert out["intent"] == "hybrid"


def test_route_node_ambiguous_short_question():
    out = route_node({"question": "who?"})
    assert out["needs_clarify"] is True


# --- 回归：回退重查必须推进状态（否则回退环无界且无效）---

def test_retrieve_node_increments_retry_count():
    chunks = [Chunk("a", "revenue grew", "d", "s", 1)]
    retrievers = {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
                  "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}
    out = retrieve_node({"question": "revenue", "intent": "semantic",
                         "retrieved_chunks": [], "retry_count": 0}, retrievers)
    assert out.get("retry_count") == 1


def test_retrieve_node_escalates_to_hybrid_on_retry():
    """回退时若沿用同一策略+同一 query，会拿到完全相同的 chunks（原地打转）。"""
    chunks = [Chunk("a", "revenue grew", "d", "s", 1)]
    seen = []

    def dense(q, k):
        seen.append(("dense", k))
        return [RetrievedChunk(chunks[0], 0.9)]

    def bm25(q, k):
        seen.append(("bm25", k))
        return [RetrievedChunk(chunks[0], 0.5)]

    retrievers = {"dense": dense, "bm25": bm25}
    first = retrieve_node({"question": "revenue", "intent": "semantic",
                           "retrieved_chunks": [], "retry_count": 0}, retrievers)
    assert [name for name, _ in seen] == ["dense"]  # 首轮：按路由走单路

    seen.clear()
    retry = retrieve_node({"question": "revenue", "intent": "semantic",
                           "retrieved_chunks": first["retrieved_chunks"],
                           "retry_count": first["retry_count"]}, retrievers)
    assert [name for name, _ in seen] == ["dense", "bm25"]  # 回退：换混合双路
    assert retry["retry_count"] == 2


# --- 回归：verify 必须落地 final_answer（成功定稿 / 超限降级）---

def test_verify_node_writes_final_answer_when_supported():
    state = {"candidate_answer": "revenue grew {{abc123}}",
             "citations": ["abc123"],
             "retrieved_chunks": [{"chunk_id": "abc123"}],
             "retry_count": 0}
    out = verify_node(state)
    assert out["grounding_verdict"] == "supported"
    assert out.get("final_answer") == "revenue grew {{abc123}}"


def test_verify_node_degrades_when_retries_exhausted():
    state = {"candidate_answer": "revenue grew {{deadbeef}}",
             "citations": ["deadbeef"],
             "retrieved_chunks": [{"chunk_id": "abc123"}],
             "retry_count": 2}  # 默认上限 2，已耗尽
    out = verify_node(state)
    assert out["grounding_verdict"] == "unsupported"
    assert out.get("final_answer")  # 不能留空、更不能抛异常


def test_verify_node_does_not_finalize_while_retries_remain():
    state = {"candidate_answer": "revenue grew {{deadbeef}}",
            "citations": ["deadbeef"],
            "retrieved_chunks": [{"chunk_id": "abc123"}],
            "retry_count": 0}
    out = verify_node(state)
    assert out["grounding_verdict"] == "unsupported"
    assert not out.get("final_answer")  # 还有重试机会，交给回退环再查


# --- CRAG：检索评估器必须真的判相关性，而不是「非空即 correct」---

class _VerdictLLM:
    def __init__(self, verdict: str):
        self.verdict = verdict

    def complete(self, prompt: str) -> str:
        return self.verdict


def _critique_state(*scores):
    return {"question": "did revenue grow?",
            "retrieved_chunks": [{"chunk_id": f"c{i}", "text": "some text", "score": s}
                                 for i, s in enumerate(scores)]}


def test_critique_marks_incorrect_when_nothing_retrieved():
    assert critique_node({"question": "q", "retrieved_chunks": []})["retrieval_verdict"] == "incorrect"


def test_critique_uses_llm_judge_verdict():
    for verdict in ("correct", "ambiguous", "incorrect"):
        out = critique_node(_critique_state(0.9), llm=_VerdictLLM(verdict))
        assert out["retrieval_verdict"] == verdict


def test_critique_reads_incorrect_before_correct():
    # "incorrect" 里包含子串 "correct"，解析顺序错了就会把不相关判成相关
    out = critique_node(_critique_state(0.9), llm=_VerdictLLM("Incorrect."))
    assert out["retrieval_verdict"] == "incorrect"


def test_critique_without_judge_never_claims_correct():
    """检索分数不能当相关性判据（BM25 小语料下真实命中也会是 0 分）。"""
    assert critique_node(_critique_state(0.9))["retrieval_verdict"] == "ambiguous"
    assert critique_node(_critique_state(0.0))["retrieval_verdict"] == "ambiguous"
    assert critique_node(_critique_state())["retrieval_verdict"] == "incorrect"


def test_critique_falls_back_when_llm_output_is_not_a_verdict():
    out = critique_node(_critique_state(0.9), llm=_VerdictLLM("我觉得还行"))
    assert out["retrieval_verdict"] == "ambiguous"   # 解析失败也不冒充 correct


def test_give_up_node_refuses_instead_of_answering():
    out = give_up_node({"question": "q", "retry_count": 2, "retrieved_chunks": []})
    assert out["grounding_verdict"] == "unsupported"
    assert out["final_answer"]
    assert out["citations"] == []


def test_give_up_names_the_rewritten_query_for_attribution():
    """改写后检不到时，拒答文案要写清用的是哪个 query —— 否则无法区分「语料没有」和「改写跑偏」。"""
    out = give_up_node({"question": "它有几幕？", "retry_count": 2,
                        "retrieved_chunks": [], "rewritten": True})
    assert "它有几幕？" in out["final_answer"]


def test_give_up_stays_clean_for_single_turn():
    out = give_up_node({"question": "q", "retry_count": 2, "retrieved_chunks": []})
    assert "改写" not in out["final_answer"]


# --- 多轮对话：指代消解（改写 query 去检索）+ 历史只进生成 prompt ---

class _BoomLLM:
    def complete(self, prompt: str) -> str:
        raise AssertionError("没有历史时不该调用 LLM 改写")


def test_rewrite_node_is_noop_without_history():
    out = rewrite_node({"question": "Who invented the telephone?", "history": []},
                       llm=_BoomLLM())
    assert out == {}


def test_rewrite_node_rewrites_when_history_present():
    class _LLM:
        def complete(self, prompt: str) -> str:
            return "Who directed Inception?"

    out = rewrite_node(
        {"question": "它的导演是谁？",
         "history": [{"question": "What is Inception?", "answer": "A 2010 film."}]},
        llm=_LLM())
    assert out["question"] == "Who directed Inception?"
    assert out["rewritten"] is True


def test_rewrite_falls_back_to_original_when_llm_returns_nothing():
    class _LLM:
        def complete(self, prompt: str) -> str:
            return "   "

    out = rewrite_node(
        {"question": "它的导演是谁？", "history": [{"question": "q", "answer": "a"}]},
        llm=_LLM())
    assert out.get("question", "它的导演是谁？") == "它的导演是谁？"   # 不能把问题改没了


def test_generate_prompt_includes_history_but_retrieval_does_not():
    prompts = []

    class _LLM:
        def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return "answer {{a}}"

    generate_node({"question": "它的导演是谁？",
                   "retrieved_chunks": [{"chunk_id": "a", "text": "Inception was directed by Nolan"}],
                   "history": [{"question": "What is Inception?", "answer": "A 2010 film."}]},
                  llm=_LLM())
    assert "What is Inception?" in prompts[0]      # 历史进生成
    assert "A 2010 film." in prompts[0]


# --- 生产链路接入统一重排（可选，缺省行为必须完全不变）---

class _FakeReranker:
    def __init__(self, prefer="gold"):
        self.prefer = prefer
        self.seen = None

    def rerank(self, query, retrieved, top_k):
        self.seen = [rc.chunk.chunk_id for rc in retrieved]
        ordered = sorted(retrieved, key=lambda rc: rc.chunk.chunk_id != self.prefer)
        return ordered[:top_k]


def _retrievers_recording():
    chunks = [Chunk("noise", "irrelevant", "d", "s", 1),
              Chunk("gold", "revenue grew 20%", "d", "s", 2)]
    calls = []

    def dense(q, k):
        calls.append(("dense", k))
        return [RetrievedChunk(chunks[0], 0.9), RetrievedChunk(chunks[1], 0.5)]

    def bm25(q, k):
        calls.append(("bm25", k))
        return [RetrievedChunk(chunks[0], 0.9)]

    return {"dense": dense, "bm25": bm25}, calls


def test_retrieve_node_reranks_when_reranker_given():
    retrievers, _ = _retrievers_recording()
    reranker = _FakeReranker("gold")
    out = retrieve_node({"question": "q", "intent": "hybrid",
                         "retrieved_chunks": [], "retry_count": 0},
                        retrievers, reranker=reranker)
    assert out["retrieved_chunks"][0]["chunk_id"] == "gold"   # 精排把它提到了第一
    assert "gold" in reranker.seen                            # 精排看得到它


def test_retrieve_node_recalls_deeper_when_reranking():
    retrievers, calls = _retrievers_recording()
    retrieve_node({"question": "q", "intent": "hybrid",
                   "retrieved_chunks": [], "retry_count": 0},
                  retrievers, reranker=_FakeReranker(), recall_k=20)
    assert all(k == 20 for _, k in calls)     # 粗排多取，留给精排挑


def test_retrieve_node_behaviour_unchanged_without_reranker():
    retrievers, calls = _retrievers_recording()
    out = retrieve_node({"question": "q", "intent": "hybrid",
                         "retrieved_chunks": [], "retry_count": 0}, retrievers)
    assert all(k == 8 for _, k in calls)                       # 原行为：取 8
    assert out["retrieved_chunks"][0]["chunk_id"] == "noise"    # 不重排，保持原顺序
