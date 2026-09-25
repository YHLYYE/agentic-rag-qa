"""top-K 扫描：{5,10,20,50} 各自测 context_recall / context_precision。"""
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from graph.nodes import route_node
from eval.ragas_self import context_precision, context_recall


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


def _retrieve(q, strategy, dense, bm25, k):
    if strategy == "keyword":
        return bm25.retrieve(q, k)
    if strategy == "hybrid":
        return merge_and_rerank([dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k)
    return dense.retrieve(q, k)


def main(n: int = 60) -> None:
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

    TOP_KS = [5, 10, 20, 50]
    results = {k: {"p": [], "r": []} for k in TOP_KS}
    for item in eval_set:
        q = item["question"]
        gt = item["ground_truth"]
        strategy = route_node({"question": q}, llm=llm)["intent"]  # 路由只做一次
        for k in TOP_KS:
            rc = _retrieve(q, strategy, dense, bm25, k)
            ctx = "\n".join(x.chunk.text for x in rc)
            results[k]["p"].append(context_precision(client, q, ctx))
            results[k]["r"].append(context_recall(client, q, gt, ctx))

    print(f"=== top-K 扫描 (n={n}) ===", flush=True)
    for k in TOP_KS:
        p = sum(results[k]["p"]) / len(results[k]["p"])
        r = sum(results[k]["r"]) / len(results[k]["r"])
        print(f"top-{k}: precision={p:.3f}  recall={r:.3f}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
