"""对比「无重排 vs bge-reranker」在 QA 语料上的 RAGAS（重点看 precision）。"""
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from rag.reranker import Reranker
from graph.nodes import route_node
from eval.ragas_self import faithfulness, context_precision, context_recall


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


def main(n: int = 90) -> None:
    load_dotenv()
    client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"], api_key=os.environ["DEEPSEEK_API_KEY"])
    llm = LLM(client)

    with open("data/qa/eval_set.pkl", "rb") as f:
        eval_set = pickle.load(f)[:n]
    with open("data/qa/index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, "data/qa/index/faiss.index")
    bm25 = BM25Retriever(chunks)
    print("加载 reranker ...", flush=True)
    reranker = Reranker()

    results = {v: {"p": [], "r": [], "f": []} for v in ("no_rerank", "rerank")}
    for item in eval_set:
        q = item["question"]
        strategy = route_node({"question": q}, llm=llm)["intent"]
        if strategy == "keyword":
            top20 = bm25.retrieve(q, 20)
        elif strategy == "hybrid":
            top20 = merge_and_rerank([dense.retrieve(q, 20), bm25.retrieve(q, 20)], top_k=20)
        else:
            top20 = dense.retrieve(q, 20)
        for variant in ("no_rerank", "rerank"):
            rc = top20[:5] if variant == "no_rerank" else reranker.rerank(q, top20, top_k=5)
            ctx = "\n".join(x.chunk.text for x in rc)
            ans = llm.complete(f"根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}")
            results[variant]["p"].append(context_precision(client, q, ctx))
            results[variant]["r"].append(context_recall(client, q, ctx, item["ground_truth"]))
            results[variant]["f"].append(faithfulness(client, q, ans, ctx))

    print(f"=== 重排对比 (n={n}) ===", flush=True)
    for v in ("no_rerank", "rerank"):
        m = results[v]
        print(f"{v}: precision={sum(m['p'])/len(m['p']):.3f} recall={sum(m['r'])/len(m['r']):.3f} faithfulness={sum(m['f'])/len(m['f']):.3f}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 90)
