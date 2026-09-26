"""Streamlit 演示：走**完整 LangGraph 链路**（路由 → 多路检索 → CRAG 批判 → 生成 → 引用硬闸门）。

此前这个界面走的是自己写的一套简化三步链路（路由 → 检索 → 生成），
没有 CRAG 自纠错、也没有引用硬闸门 —— 和图是两套实现。
现在改成复用 `run_graph` 的同一套执行与追踪逻辑，消除这份结构性冗余。

运行：streamlit run src/ui/app.py
"""
import os
import sys
from pathlib import Path

# 让 `streamlit run src/ui/app.py` 无需手动设 PYTHONPATH 就能导入 src 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 国内走 hf-mirror 镜像，避免连 huggingface.co 卡住
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import streamlit as st

import run_graph

@st.cache_resource
def load_retrievers(corpus_name: str = "en_qa") -> dict:
    """按语料加载检索器 —— 索引目录与分词器都由语料档案决定（分词器用错会让 BM25 失效）。"""
    corpus = run_graph.CORPORA[corpus_name]
    # device="cpu"：界面与显存敏感，强制 CPU（与 run_graph 的演示路径一致）
    return run_graph.load_retrievers(corpus["index_dir"], device="cpu",
                                     tokenizer=corpus["tokenizer"])


@st.cache_resource
def load_llm():
    return run_graph.build_llm(degrade=True)   # 打不通就降级，别让界面 500


@st.cache_resource
def load_reranker():
    """加载失败返回 None —— `run_graph.build_reranker` 内部已做优雅降级，不让界面崩。"""
    return run_graph.build_reranker()


@st.cache_resource
def load_graph(corpus_name: str = "en_qa", use_rerank: bool = False):
    """编译**带 checkpointer** 的图：这样 interrupt() 是「可恢复的暂停」而不是「只能重跑」。

    cache_resource 保证同一次会话里拿到的是**同一个图实例**（恢复依赖它的存档）。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from graph.build import build_graph

    return build_graph(load_retrievers(corpus_name), load_llm(),
                       reranker=load_reranker() if use_rerank else None,
                       checkpointer=MemorySaver())


def render_state(state: dict, show_answer: bool = False) -> None:
    """渲染链路追踪 + **可溯源的引用对照表**。

    答案里的 `{{chunk_id}}`（给机器校验用）会换成 `[1] [2]`，与下面的来源编号一一对应，
    每条来源都带 source_doc / 页码 / 章节 —— 这就是「引用可溯源」。
    """
    rendered_answer, sources = run_graph.number_citations(
        state.get("final_answer") or "", state.get("retrieved_chunks") or [])

    with st.container(border=True):
        st.markdown("**链路追踪**")
        st.text("\n".join(run_graph.format_trace(state)))

    if show_answer:
        st.subheader("答案")
        st.markdown(rendered_answer or "（没有产出答案）")

    st.subheader(f"引用来源（{len(sources)} 条）")
    if not sources:
        st.caption("本条答案没有引用（例如拒答）。")
    for src in sources:
        label = f"[{src['n']}] {src.get('source_doc') or src['chunk_id']}"
        if src.get("page"):
            label += f" · 第 {src['page']} 页"
        if src.get("section"):
            label += f" · {src['section']}"
        with st.expander(label):
            st.caption(f"chunk_id: {src['chunk_id']}")
            st.write(src["text"])


st.set_page_config(page_title="Agentic RAG 问答", layout="wide")
st.title("Agentic RAG 问答")
st.caption("自纠错 + 引用硬闸门：路由 → 多路检索 → CRAG 批判 → 生成 → 引用校验")

# 会话状态：thread_id 用于 checkpoint（澄清恢复要靠它），pending 存被中断的那次请求
st.session_state.setdefault("thread_id", "web")
st.session_state.setdefault("pending", None)
st.session_state.setdefault("pending_cfg", None)
st.session_state.setdefault("messages", [])      # 展示用的对话记录
st.session_state.setdefault("rag_history", [])   # 传给图的多轮历史（只进生成 prompt）

# 语料与精排放在侧栏：chat_input 不能放在 form 里，所以把它们移出表单
with st.sidebar:
    corpus_name = st.selectbox(
        "语料",
        options=sorted(run_graph.CORPORA),
        format_func=lambda k: run_graph.CORPORA[k]["label"],
    )
    use_rerank = st.toggle("启用统一重排（更准，但延迟约翻倍）", value=False)
    if not run_graph.CORPORA[corpus_name]["has_answers"]:
        st.info("该语料没有标准答案：可以看检索与引用，但不能据此报告「答案正确率」。")
    if use_rerank and load_reranker() is None:
        st.warning("精排模型加载失败（多为内存不足），本次已降级为不精排运行。")

# 渲染既有对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 多轮输入。历史会喂给 rewrite_node 做指代消解（"它的导演是谁" 能被正确改写）
question = st.chat_input("问一个问题，可追问（如「它的导演是谁？」）")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    graph = load_graph(corpus_name, use_rerank)
    rendered_state = None
    with st.chat_message("assistant"):
        with st.spinner("检索与生成中..."):
            state = run_graph.run(question, graph=graph,
                                  thread_id=st.session_state.thread_id,
                                  history=st.session_state.rag_history)
        if "__interrupt__" in state:
            st.warning("问题过于模糊，已在澄清节点暂停。请在下方补充信息，我会**从暂停处继续**。")
            st.session_state.pending = state
            st.session_state.pending_cfg = (corpus_name, use_rerank)
        else:
            answer = state.get("final_answer") or "（没有产出答案）"
            # 气泡里显示**编号后**的答案（[1] [2] 与下方来源对照表一一对应）
            rendered_answer, _ = run_graph.number_citations(
                answer, state.get("retrieved_chunks") or [])
            st.markdown(rendered_answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.rag_history.append({"question": question, "answer": answer})
            rendered_state = state
    # 开发视图（链路追踪 + 引用来源）放在气泡**外面**折叠显示，保持对话区干净
    if rendered_state is not None:
        with st.expander("链路追踪与引用来源", expanded=True):
            render_state(rendered_state)

# 澄清态：显示追问输入，补充后从 interrupt 处**续跑**（不是重新提问）
if st.session_state.pending:
    st.warning("问题过于模糊，链路已在 clarify 节点暂停 —— 补充信息后会**从暂停处继续**，不会重跑。")
    with st.form("clarify", border=True):
        extra = st.text_input("请补充你的问题", placeholder="例如：Who invented the telephone?")
        go = st.form_submit_button("继续", icon=":material/play_arrow:")
    if go and extra.strip():
        corpus_name, use_rerank = st.session_state.pending_cfg
        state = run_graph.resume(extra, load_graph(corpus_name, use_rerank),
                                 st.session_state.thread_id)
        st.session_state.pending = None
        st.success("已从暂停处续跑完成。")
        render_state(state, show_answer=True)
        if state.get("final_answer"):
            st.session_state.messages.append({"role": "assistant", "content": state["final_answer"]})
            st.session_state.rag_history.append({"question": extra, "answer": state["final_answer"]})
