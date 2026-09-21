"""Three-group retrieval evaluation over FinanceBench (MRR / HitRate@5)."""
import pickle
from pathlib import Path

from datasets import load_dataset

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from graph.nodes import route_node
from eval.metrics import mrr, hit_rate


def _ranked_docs(retrieved) -> list[str]:
    docs = []
    for x in retrieved:
        d = x.chunk.source_doc
        if d not in docs:
            docs.append(d)
    return docs


def main(index_dir: str = "data/index", n: int = 150) -> None:
    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"loaded {len(chunks)} chunks", flush=True)

    print("loading bge-m3 ...", flush=True)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)

    ds = load_dataset("PatronusAI/financebench", split="train")
    questions = list(ds["question"][:n])
    doc_names = list(ds["doc_name"][:n])

    groups = {
        "1_naive_dense": lambda q: dense.retrieve(q, top_k=10),
        "2_hybrid": lambda q: merge_and_rerank(
            [dense.retrieve(q, 10), bm25.retrieve(q, 10)], top_k=10),
        "3_routed": lambda q: (bm25.retrieve(q, 10)
                               if route_node({"question": q})["intent"] == "keyword"
                               else dense.retrieve(q, 10)),
    }

    report = {}
    for name, retrieve in groups.items():
        rankings, gts = [], []
        for i, q in enumerate(questions):
            rc = retrieve(q)
            rankings.append(_ranked_docs(rc))
            gts.append({doc_names[i]})
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i+1}/{n}", flush=True)
        report[name] = {
            "mrr": round(mrr(rankings, gts), 4),
            "hit_rate@5": round(hit_rate(rankings, gts, k=5), 4),
        }

    print("\n=== 三组检索对照 (MRR / HitRate@5) ===", flush=True)
    for name, m in report.items():
        print(f"{name}: MRR={m['mrr']}  HitRate@5={m['hit_rate@5']}", flush=True)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
