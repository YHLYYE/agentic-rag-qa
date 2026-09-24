"""Streamlit 演示界面：问 → 路由(LLM意图分类) → 检索 → 带引用回答。"""
import os
import pickle
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from graph.nodes import route_node


class LLM:
    def __init__(self, client: OpenAI):
        self.client = client

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content.strip()


@st.cache_resource
def load_retrievers(index_dir="data/qa/index"):
    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)
    return chunks, dense, bm25


@st.cache_resource
def load_llm():
    load_dotenv()
    client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"], api_key=os.environ["DEEPSEEK_API_KEY"])
    return LLM(client)


_STRATEGY_LABEL = {
    "keyword": "BM25（关键词）",
    "semantic": "dense 向量（语义）",
    "hybrid": "混合（dense+BM25）",
}


def main():
    st.set_page_config(page_title="Agentic RAG 问答", layout="wide")
    st.title("🤖 Agentic RAG 问答")
    st.caption("自纠错 + 引用硬闸门：先路由分类，再检索，答案必须引用真实资料")

    load_dotenv()
    question = st.text_input("问一个问题", placeholder="例如：谁发明了电话？/ A 和 B 哪个更高？/ A 主演的电影的导演是谁？")

    if question:
        chunks, dense, bm25 = load_retrievers()
        llm = load_llm()

        # 1. 路由
        out = route_node({"question": question}, llm=llm)
        intent = out["route_decision"]["semantic_intent"]
        strategy = out["intent"]
        st.info(f"🔀 路由：{intent} → {_STRATEGY_LABEL.get(strategy, strategy)}")

        # 2. 检索
        with st.spinner("检索中..."):
            if strategy == "keyword":
                top5 = bm25.retrieve(question, 5)
            elif strategy == "hybrid":
                top5 = merge_and_rerank([dense.retrieve(question, 10), bm25.retrieve(question, 10)], top_k=5)
            else:
                top5 = dense.retrieve(question, 5)
        ctx = "\n\n".join(f"[{x.chunk.source_doc}] {x.chunk.text}" for x in top5)

        # 3. 生成
        with st.spinner("生成答案中..."):
            answer = llm.complete(
                f"根据以下上下文回答问题，尽量简洁，并标注引用了哪份资料。"
                f"如果上下文查不到，就回答「查不到」。\n\n问题：{question}\n\n上下文：\n{ctx}"
            )

        st.subheader("答案")
        st.markdown(answer)

        st.subheader("引用来源（检索到的资料）")
        for i, x in enumerate(top5, 1):
            with st.expander(f"{i}. {x.chunk.source_doc}"):
                st.write(x.chunk.text[:400])


if __name__ == "__main__":
    main()
