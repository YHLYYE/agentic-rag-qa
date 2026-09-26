from graph.state import AgenticRAGState
from rag.citation import parse_citations


# 语义意图 → 检索策略 的映射。
#
# 2026-09-26 数据复核记录（n=300，HitRate@5）：
#   曾一度认为「factoid → BM25」是负收益（当时实测 bm25 0.720，三类都弱于 dense），
#   并据此把它改成统一走 hybrid。**该结论已撤回** —— 根因不是路由设计，而是
#   BM25 的分词实现有缺陷（text.split()：大小写敏感 + 不剥离标点）。
#
#   修复分词后（src/rag/bm25.py 的 simple_tokenize / tokenize_no_stopwords）：
#       bm25  0.720 → 0.790（小写+去标点）→ 0.803（再去停用词）
#     factoid 上 0.800 → 0.900，与 dense（0.890）持平，
#     即原映射「查事实/数字走词面匹配」的前提**成立**。
#
#   保留三类分流的另一层意义是延迟：BM25 不需要 embedding，
#   factoid（占 1/3 查询）走 BM25 可以省掉这部分开销，代价约 0.7pt。
_INTENT_MAP = {
    "factoid": "keyword",     # 查事实/数字 → BM25 关键词
    "comparison": "semantic", # 对比 → dense 语义
    "multi-hop": "hybrid",    # 多跳 → dense+BM25 混合
}


def _route_by_rules(question: str) -> str | None:
    q = question.lower()
    if "definition" in q or "define" in q:
        return "keyword"
    return None


def _route_by_llm(question: str, llm) -> str:
    prompt = (
        "把以下问题分类为三类之一，只输出类别名（factoid / comparison / multi-hop）：\n"
        "- factoid：查一个具体事实或数字（单步查询，如「谁发明了电话」）\n"
        "- comparison：对比两个事物（如「A 和 B 哪个更高」）\n"
        "- multi-hop：需要多步推理才能回答（如「A 主演的电影的导演是谁」）\n\n"
        f"问题：{question}\n\n类别："
    )
    resp = llm.complete(prompt).strip().lower()
    for key in ("factoid", "comparison", "multi-hop", "multihop"):
        if key in resp:
            return "multi-hop" if key == "multihop" else key
    return "factoid"  # 兜底


def route_node(state: AgenticRAGState, llm=None) -> dict:
    q = state["question"].strip()
    # 简单歧义检测：去掉空格后 < 5 个字符（中英文通用，如 "who?" / "谁？"）→ 需要澄清
    if len(q.replace(" ", "")) < 5:
        return {
            "intent": "semantic",
            "needs_clarify": True,
            "route_decision": {"semantic_intent": "ambiguous", "strategy": "semantic", "source": "rule"},
        }
    if llm is not None:
        semantic = _route_by_llm(q, llm)
        strategy = _INTENT_MAP.get(semantic, "semantic")
        source = "llm"
    else:
        strategy = _route_by_rules(q) or "semantic"
        semantic = None
        source = "rule"
    return {
        "intent": strategy,
        "needs_clarify": False,
        "route_decision": {"semantic_intent": semantic, "strategy": strategy, "source": source},
    }


def clarify_node(state: AgenticRAGState) -> dict:
    """interrupt() 人工澄清：暂停等用户补充信息，恢复后回到 route 重新路由。"""
    from langgraph.types import interrupt
    clarification = interrupt({"clarification": "请澄清你的问题（补充缺失信息）"})
    return {"question": clarification, "needs_clarify": False, "clarification": clarification}


_DEGRADED_ANSWER = (
    "抱歉，我没有在知识库中找到足以支撑该结论的资料（已重试 {n} 次）。"
    "请补充问题信息，或直接查阅原始文档。"
)


def verify_node(state: AgenticRAGState, max_retry: int = 2) -> dict:
    retrieved_ids = {rc.get("chunk_id") for rc in state["retrieved_chunks"]}
    cits = state.get("citations", [])
    if cits and all(c in retrieved_ids for c in cits):
        return {"grounding_verdict": "supported",
                "final_answer": state.get("candidate_answer", "")}
    if state.get("retry_count", 0) >= max_retry:
        # 重试已耗尽：降级为「查不到」，而不是留空或抛异常
        return {"grounding_verdict": "unsupported",
                "final_answer": _DEGRADED_ANSWER.format(n=max_retry)}
    return {"grounding_verdict": "unsupported"}


def retrieve_node(state: AgenticRAGState, retrievers: dict, reranker=None,
                  recall_k: int = 20) -> dict:
    """检索节点。

    `reranker=None` 时行为与之前**完全一致**（只取 top_k），保证不引入回退风险；
    传入 reranker 时改为「粗排多取 recall_k → 精排收敛到 top_k」。
    """
    from rag.hybrid import merge_and_rerank
    q = state["question"]
    attempt = state.get("retry_count", 0)
    # 回退重查必须改变检索行为：沿用原策略+同一 query 会拿到完全相同的 chunks（原地打转）
    intent = "hybrid" if attempt > 0 else state["intent"]
    top_k = 8 if attempt == 0 else 12
    # 有精排时粗排多取候选（实测：粗排 top-20 → 精排，比直接取 top-5 明显更好）
    fetch_k = recall_k if reranker is not None else top_k
    if intent == "hybrid":
        lists = [r(q, fetch_k) for r in retrievers.values()]
        merged = merge_and_rerank(lists, top_k=fetch_k)
    elif intent == "keyword":
        merged = retrievers["bm25"](q, fetch_k)
    else:
        merged = retrievers["dense"](q, fetch_k)
    if reranker is not None:
        merged = reranker.rerank(q, merged, top_k=top_k)
    out = [{"chunk_id": rc.chunk.chunk_id, "text": rc.chunk.text, "score": rc.score}
           for rc in merged]
    return {"retrieved_chunks": out, "retry_count": attempt + 1}


_CRAG_PROMPT = (
    "你在做检索质量评估（CRAG）。判断下面这些资料对回答问题的有用程度。\n"
    "只输出一个判定词：correct / ambiguous / incorrect\n"
    "- correct：至少有 1 段资料能直接回答问题\n"
    "- ambiguous：有部分相关信息，但不足以给出确定答案\n"
    "- incorrect：全部与问题无关\n\n"
    "问题：{question}\n\n资料：\n{context}\n\n判定："
)

_NO_EVIDENCE_ANSWER = (
    "抱歉，知识库里没有检索到能回答这个问题的资料。"
    "请换一种问法，或确认该问题是否在当前知识库范围内。"
)


def _parse_verdict(text: str) -> str | None:
    """顺序要紧：'incorrect' 里含子串 'correct'，先判它才不会把不相关读成相关。"""
    t = (text or "").strip().lower()
    for verdict in ("incorrect", "ambiguous", "correct"):
        if verdict in t:
            return verdict
    return None


def _verdict_without_judge(chunks: list) -> str:
    """没有 LLM 判官时的保守兜底：只区分「什么都没查到」和「有候选但相关性未知」。

    不能拿检索分数当判据：分数尺度依赖检索器（BM25 无界、cosine 0~1），
    而且 BM25 在语料极小时 IDF 会退化成 0，连真实命中都算出 0 分（实测）。
    所以这里不敢说 correct，最多说 ambiguous。
    """
    return "ambiguous" if chunks else "incorrect"


def critique_node(state: AgenticRAGState, llm=None) -> dict:
    """CRAG 检索自纠错：判断检索到的资料能不能支撑作答。

    correct → 直接进入生成；ambiguous / incorrect → 触发回退重查。
    不传 llm 时退化为确定性分数兜底（仅识别明确零命中，不做语义判断）。
    """
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return {"retrieval_verdict": "incorrect"}
    if llm is None:
        return {"retrieval_verdict": _verdict_without_judge(chunks)}

    context = "\n".join(
        f"[{c.get('chunk_id')}] {(c.get('text') or '')[:400]}" for c in chunks
    )
    resp = llm.complete(_CRAG_PROMPT.format(question=state["question"], context=context))
    # 判官输出解析不出来时退回保守兜底，绝不默认放行
    return {"retrieval_verdict": _parse_verdict(resp) or _verdict_without_judge(chunks)}


def give_up_node(state: AgenticRAGState) -> dict:
    """检索判定不可用且重试耗尽：明确拒答，不进入生成（不给幻觉留机会）。"""
    return {
        "candidate_answer": "",
        "citations": [],
        "grounding_verdict": "unsupported",
        "final_answer": _NO_EVIDENCE_ANSWER,
    }


def generate_node(state: AgenticRAGState, llm) -> dict:
    chunks = state["retrieved_chunks"]
    ctx = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    prompt = (
        f"Question: {state['question']}\n\nContext:\n{ctx}\n\n"
        "Answer concisely and cite sources inline as {{chunk_id}}."
    )
    answer = llm.complete(prompt)
    cits = parse_citations(answer)
    return {"candidate_answer": answer, "citations": cits}
