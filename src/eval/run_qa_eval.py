"""对 HotpotQA + TriviaQA 语料跑 RAGAS（faithfulness / precision / recall），分题型统计。"""
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
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


def main(n: int = 300) -> None:
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

    results = {t: {"f": [], "p": [], "r": []} for t in ("factoid", "comparison", "multi-hop")}
    for item in eval_set:
        q = item["question"]
        strategy = route_node({"question": q}, llm=llm)["intent"]
        if strategy == "keyword":
            rc = bm25.retrieve(q, 5)
        elif strategy == "hybrid":
            rc = merge_and_rerank([dense.retrieve(q, 10), bm25.retrieve(q, 10)], top_k=5)
        else:
            rc = dense.retrieve(q, 5)
        ctx = "\n".join(x.chunk.text for x in rc)
        ans = llm.complete(f"根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}")
        t = item["type"]
        results[t]["f"].append(faithfulness(client, q, ans, ctx))
        results[t]["p"].append(context_precision(client, q, ctx))
        results[t]["r"].append(context_recall(client, q, ctx, item["ground_truth"]))

    print("=== RAGAS 分题型 ===", flush=True)
    all_f, all_p, all_r = [], [], []
    for t, m in results.items():
        if m["f"]:
            f = sum(m["f"]) / len(m["f"])
            p = sum(m["p"]) / len(m["p"])
            r = sum(m["r"]) / len(m["r"])
            all_f += m["f"]; all_p += m["p"]; all_r += m["r"]
            print(f"{t}: faithfulness={f:.3f} precision={p:.3f} recall={r:.3f}", flush=True)
    print(f"整体: faithfulness={sum(all_f)/len(all_f):.3f} precision={sum(all_p)/len(all_p):.3f} recall={sum(all_r)/len(all_r):.3f}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
