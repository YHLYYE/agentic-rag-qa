"""确定性检索指标（不需要 LLM 判官）：每轮优化的主信号。

为什么单独一套：LLM 判官（recall / faithfulness / context_precision）既花钱又有噪声，
不适合做快速迭代的标尺。这里用「检索到的 chunk 里是否含标准答案」这种可精确匹配的
事实来计算 HitRate / MRR / 答案占比，纯确定性、可复现、可跨版本对比。

已知局限：
1. 字符串精确匹配是**代理指标**：可能漏掉改写表述，factoid 的短答案也可能偶然命中。
2. 它衡量「检没检到、排第几」，**不衡量答案好不好**——生成质量仍要看 faithfulness。
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

from eval import artifacts


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def answer_in_text(ground_truth: str, text: str) -> bool:
    gt = normalize(ground_truth)
    return bool(gt) and gt in normalize(text)


def _first_hit_rank(chunks, ground_truth: str) -> int | None:
    for i, rc in enumerate(chunks, 1):
        if answer_in_text(ground_truth, rc.chunk.text):
            return i
    return None


def evaluate_retrieval(retrieve_fn, eval_set: list[dict], k: int = 5) -> dict:
    """retrieve_fn(question, k) -> list[RetrievedChunk]。返回汇总 + 逐题明细。"""
    rows = []
    for item in eval_set:
        chunks = retrieve_fn(item["question"], k)
        hits = [i for i, rc in enumerate(chunks, 1)
                if answer_in_text(item["ground_truth"], rc.chunk.text)]
        rows.append({
            "question": item["question"],
            "type": item["type"],
            "ground_truth": item["ground_truth"],
            "retrieved_chunk_ids": [rc.chunk.chunk_id for rc in chunks],
            "hit": bool(hits),
            "first_hit_rank": hits[0] if hits else None,
            "hits_in_topk": len(hits),
            "answer_ratio": len(hits) / k if chunks else 0.0,
        })
    return _summarize(rows, k, detail_rows=rows)


def _summarize(rows: list[dict], k: int, detail_rows: list[dict]) -> dict:
    n = len(rows)
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    def _block(sub: list[dict]) -> dict:
        hits = [r for r in sub if r["hit"]]
        ranks = [r["first_hit_rank"] for r in hits]
        return {
            "n": len(sub),
            "hit_rate@k": len(hits) / len(sub) if sub else 0.0,
            "mrr": sum(1.0 / x for x in ranks) / len(sub) if sub else 0.0,
            "answer_ratio@k": sum(r["answer_ratio"] for r in sub) / len(sub) if sub else 0.0,
        }

    summary = {
        "k": k,
        **_block(rows),
        "miss_count": n - sum(1 for r in rows if r["hit"]),
        "first_hit_rank": {str(rank): c for rank, c in
                           sorted(collections.Counter(
                               r["first_hit_rank"] for r in rows if r["hit"]).items())},
        "per_type": {t: _block(v) for t, v in sorted(by_type.items())},
        "rows": detail_rows,
    }
    return summary


def load_eval_set(path: str = "data/qa/eval_set.pkl") -> list[dict]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def bench_latency(retrieve_fn, questions: list[str], k: int = 5,
                  warmup: int = 3) -> dict:
    """测单次检索延迟（毫秒）。warmup 先跑几次，避免把首次加载/缓存冷启动算进去。

    这是「路由策略平衡精度延迟」的前提：没有 p50/p95 就没法做延迟决策。
    """
    import statistics
    import time

    for q in questions[:warmup]:
        retrieve_fn(q, k)
    times_ms = []
    for q in questions:
        t0 = time.perf_counter()
        retrieve_fn(q, k)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()

    def _pct(p: float) -> float:
        idx = min(len(times_ms) - 1, int(round(p / 100 * len(times_ms))) - 1)
        return times_ms[max(0, idx)]

    return {
        "n": len(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "p50_ms": _pct(50),
        "p95_ms": _pct(95),
        "max_ms": times_ms[-1],
    }


def load_chunks(index_dir: str = "data/qa/index"):
    import pickle
    from pathlib import Path
    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        return pickle.load(f)


def build_retriever(kind: str, index_dir: str = "data/qa/index", device: str = "cpu"):
    """kind: dense | bm25 | bm25_split | bm25_nostop | hybrid | hybrid_rerank。"""
    from rag.bm25 import BM25Retriever
    from rag.dense import DenseRetriever
    from rag.embeddings import Embedder
    from rag.hybrid import merge_and_rerank

    index_path = Path(index_dir)
    chunks = load_chunks(index_dir)
    if kind == "bm25_split":                      # 旧的缺陷实现（大小写/标点敏感）
        bm25 = BM25Retriever(chunks, tokenize=lambda t: (t or "").split())
    elif kind == "bm25_nostop":
        from rag.bm25 import tokenize_no_stopwords
        bm25 = BM25Retriever(chunks, tokenize=tokenize_no_stopwords)
    else:
        bm25 = BM25Retriever(chunks)              # 默认 simple_tokenize
    if kind in ("bm25", "bm25_split", "bm25_nostop"):
        return lambda q, k: bm25.retrieve(q, k)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024, device=device,
                        show_progress=False)
    dense = DenseRetriever(chunks, embedder, str(index_path / "faiss.index"))
    if kind == "dense":
        return lambda q, k: dense.retrieve(q, k)
    if kind == "hybrid":
        return lambda q, k: merge_and_rerank(
            [dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k)
    if kind == "hybrid_rerank":
        from rag.pipeline import HybridRerankRetriever
        from rag.reranker import Reranker
        pipe = HybridRerankRetriever(dense, bm25, Reranker(),
                                     recall_k=20, rerank_k=5)
        return lambda q, k: pipe.retrieve(q, k)
    raise ValueError(f"unknown retriever kind: {kind}")


def main(retriever: str = "dense", k: int = 5, per_type: int | None = None,
         limit: int | None = None, index_dir: str = "data/qa/index",
         out_dir: str = artifacts.DEFAULT_OUT_DIR, bench: int = 0) -> dict:
    from pathlib import Path

    eval_set = load_eval_set()
    if per_type:
        from eval import sampling
        eval_set = sampling.stratified_sample(eval_set, per_type)
    elif limit:
        eval_set = eval_set[:limit]

    print(f"检索器={retriever}  k={k}  n={len(eval_set)}  索引={index_dir}", flush=True)
    retrieve_fn = build_retriever(retriever, index_dir=index_dir)

    if bench:
        questions = [it["question"] for it in eval_set[:bench]]
        stats = bench_latency(retrieve_fn, questions, k=k)
        print(f"延迟(ms): p50={stats['p50_ms']:.1f}  p95={stats['p95_ms']:.1f}  "
              f"mean={stats['mean_ms']:.1f}  n={stats['n']}", flush=True)
        return stats

    summary = evaluate_retrieval(retrieve_fn, eval_set, k=k)

    print(f"HitRate@{k} = {summary['hit_rate@k']:.4f}   MRR = {summary['mrr']:.4f}   "
          f"答案占比 = {summary['answer_ratio@k']:.4f}   未检到 = {summary['miss_count']}/{summary['n']}",
          flush=True)
    print("首次命中排名:", summary["first_hit_rank"], flush=True)
    for t, b in summary["per_type"].items():
        print(f"  {t:<11} HitRate={b['hit_rate@k']:.3f}  MRR={b['mrr']:.3f}  "
              f"答案占比={b['answer_ratio@k']:.3f}  n={b['n']}", flush=True)

    payload = {kk: vv for kk, vv in summary.items() if kk != "rows"}
    payload["retriever"] = retriever
    payload["index_dir"] = index_dir
    paths = artifacts.save_run(f"retrieval_{retriever}_k{k}", summary["rows"], payload,
                               out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="确定性检索指标（无需 LLM）")
    p.add_argument("--retriever", default="dense",
                   choices=["dense", "bm25", "bm25_split", "bm25_nostop",
                            "hybrid", "hybrid_rerank"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--index-dir", default="data/qa/index")
    p.add_argument("--bench", type=int, default=0,
                   help="只测延迟：跑前 N 题，输出 p50/p95（不做命中率评估）")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    main(retriever=a.retriever, k=a.k, per_type=a.per_type, limit=a.limit,
         index_dir=a.index_dir, out_dir=a.out, bench=a.bench)
