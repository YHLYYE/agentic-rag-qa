"""Streamlit 演示界面：问 → 检索 → 生成 → 带引用回答。"""
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


@st.cache_resource
def load_retrievers(index_dir="data/index"):
    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)
    return chunks, dense, bm25


def main():
    st.set_page_config(page_title="财报问答", layout="wide")
    st.title("📊 财报问答（Agentic RAG）")
    st.caption("自纠错 + 引用硬闸门：答案必须引用真实资料，查不到就老实说查不到")

    load_dotenv()
    question = st.text_input("问一个财报问题", placeholder="例如：3M 公司 2018 年资本支出是多少？")

    if question:
        with st.spinner("检索中..."):
            chunks, dense, bm25 = load_retrievers()
            top20 = merge_and_rerank(
                [dense.retrieve(question, 20), bm25.retrieve(question, 20)], top_k=20)
            top5 = top20[:5]
            ctx = "\n\n".join(f"[{x.chunk.source_doc} p{x.chunk.page}] {x.chunk.text}" for x in top5)

        with st.spinner("生成答案中..."):
            client = OpenAI(
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
            resp = client.chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": (
                    f"根据以下上下文回答问题，尽量简洁，并标注你引用了哪份资料的哪一页。"
                    f"如果上下文里查不到，就回答「查不到」。\n\n问题：{question}\n\n上下文：\n{ctx}"
                )}],
                temperature=0,
            )
            answer = resp.choices[0].message.content

        st.subheader("答案")
        st.markdown(answer)

        st.subheader("引用来源（检索到的资料）")
        for i, x in enumerate(top5, 1):
            with st.expander(f"{i}. {x.chunk.source_doc} 第{x.chunk.page}页"):
                st.write(x.chunk.text[:400])


if __name__ == "__main__":
    main()
