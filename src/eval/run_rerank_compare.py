"""Compare no-rerank vs bge-reranker on RAGAS metrics (faithfulness/precision/recall)."""
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from rag.reranker import Reranker
from eval.ragas_self import faithfulness, context_precision, context_recall


def _ask(client, prompt: str) -> str:
    r = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()


def main(index_dir: str = "data/index", n: int = 15) -> None:
    load_dotenv()
    client = OpenAI(
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )

    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)
    print("loading reranker ...", flush=True)
    reranker = Reranker()

    ds = load_dataset("PatronusAI/financebench", split="train")
    questions = list(ds["question"][:n])
    answers = list(ds["answer"][:n])

    results = {v: {"f": [], "p": [], "r": []} for v in ("no_rerank", "rerank")}
    for i, (q, gt) in enumerate(zip(questions, answers), 1):
        top20 = merge_and_rerank([dense.retrieve(q, 20), bm25.retrieve(q, 20)], top_k=20)
        for variant in ("no_rerank", "rerank"):
            rc = top20[:5] if variant == "no_rerank" else reranker.rerank(q, top20, top_k=5)
            ctx = "\n".join(x.chunk.text for x in rc)
            ans = _ask(client, f"根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}")
            results[variant]["f"].append(faithfulness(client, q, ans, ctx))
            results[variant]["p"].append(context_precision(client, q, ctx))
            results[variant]["r"].append(context_recall(client, q, ctx, gt))
        print(f"[{i}/{n}] done", flush=True)

    print("\n=== 重排对比 (n=%d) ===" % n, flush=True)
    for variant in ("no_rerank", "rerank"):
        m = results[variant]
        print(f"{variant}: faithfulness={sum(m['f'])/len(m['f']):.3f}  "
              f"precision={sum(m['p'])/len(m['p']):.3f}  recall={sum(m['r'])/len(m['r']):.3f}",
              flush=True)


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(argv[0] if argv else "data/index", int(argv[1]) if len(argv) > 1 else 15)
