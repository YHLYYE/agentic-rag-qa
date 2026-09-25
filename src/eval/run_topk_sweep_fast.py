"""top-K 扫描（快速版）：用单次判断的简版 recall/precision，不用拆句法。"""
import os
import pickle

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


def _score(client, prompt: str) -> float:
    r = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(r.choices[0].message.content.strip())
    except ValueError:
        return 0.0


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
        strategy = route_node({"question": q}, llm=llm)["intent"]
        for k in TOP_KS:
            rc = _retrieve(q, strategy, dense, bm25, k)
            ctx = "\n".join(x.chunk.text for x in rc)
            results[k]["p"].append(_score(client, (
                f"给定问题和检索到的上下文，判断上下文中有多少比例的内容对回答有用（精准度）。"
                f"\n问题：{q}\n上下文：{ctx}\n只输出 0~1 小数。")))
            results[k]["r"].append(_score(client, (
                f"给定问题、检索到的上下文、标准答案，判断上下文覆盖了标准答案多少比例的信息（召回）。"
                f"\n问题：{q}\n上下文：{ctx}\n标准答案：{gt}\n只输出 0~1 小数。")))

    print(f"=== top-K 扫描（快速版, n={n}）===", flush=True)
    for k in TOP_KS:
        p = sum(results[k]["p"]) / len(results[k]["p"])
        r = sum(results[k]["r"]) / len(results[k]["r"])
        print(f"top-{k}: precision={p:.3f}  recall={r:.3f}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
