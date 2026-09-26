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
    return run_graph.build_llm()


@st.cache_resource
def load_reranker():
    """加载失败返回 None —— `run_graph.build_reranker` 内部已做优雅降级，不让界面崩。"""
    return run_graph.build_reranker()


st.set_page_config(page_title="Agentic RAG 问答", layout="wide")
st.title("Agentic RAG 问答")
st.caption("自纠错 + 引用硬闸门：路由 → 多路检索 → CRAG 批判 → 生成 → 引用校验")

# 用 form 批量提交：否则每敲一个字符都会跑一遍检索 + LLM 生成
with st.form("ask", border=False):
    question = st.text_input(
        "问一个问题",
        placeholder="例如：Which opera has more acts, La jolie fille de Perth or Mitridate, re di Ponto?",
    )
    corpus_name = st.selectbox(
        "语料",
        options=sorted(run_graph.CORPORA),
        format_func=lambda k: run_graph.CORPORA[k]["label"],
    )
    use_rerank = st.toggle("启用统一重排（更准，但延迟约翻倍）", value=False)
    submitted = st.form_submit_button("提问", icon=":material/search:")

if submitted and not question.strip():
    st.warning("请先输入问题。")
elif submitted:
    retrievers = load_retrievers(corpus_name)
    llm = load_llm()
    reranker = load_reranker() if use_rerank else None
    if not run_graph.CORPORA[corpus_name]["has_answers"]:
        st.info("该语料没有标准答案：可以看检索与引用，但不能据此报告「答案正确率」。")
    if use_rerank and reranker is None:
        st.warning("精排模型加载失败（多为内存不足），本次已降级为不精排运行。")

    # 先渲染封面与标题，把耗时的链路跑在预留容器里，避免整页卡住
    slot = st.container()
    with slot.skeleton():
        state = run_graph.run(question, retrievers, llm, reranker=reranker)

    if "__interrupt__" in state:
        st.warning("问题过于模糊，链路已在澄清节点暂停。请补充信息后重新提问。")
    else:
        with st.container(border=True):
            st.markdown("**链路追踪**")
            st.text("\n".join(run_graph.format_trace(state)))

        st.subheader("答案")
        st.markdown(state.get("final_answer") or "（没有产出答案）")

        chunks = state.get("retrieved_chunks") or []
        st.subheader(f"检索到的资料（{len(chunks)} 段）")
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"{i}. {chunk['chunk_id']}"):
                st.write(chunk["text"])
