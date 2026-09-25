"""两段式重排评估：把「粗排缓存」与「精排打分」拆到两个进程。

为什么必须拆：本机（15GB 内存、8GB 显存、页面文件受限）把 bge-m3 和 bge-reranker
同时装进一个进程会直接 `OSError 1455 页面文件太小`。拆成两段后：

    第 1 段（只加载 bge-m3）：算 dense+BM25 的 RRF 融合候选，缓存为 pkl
    第 2 段（只加载 bge-reranker）：读缓存 → CrossEncoder 精排 → 出指标

这同时也是更贴近生产的部署形态（粗排与精排可以是不同服务、不同机器）。

用法：
    python -m eval.rerank_eval --stage cache  --recall-k 20 [--per-type 20]
    python -m eval.rerank_eval --stage rerank --k 5
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from eval import artifacts, retrieval_metrics

DEFAULT_CACHE = "data/qa/hybrid_candidates.pkl"


def cache_candidates(eval_set: list[dict], retrieve_fn, recall_k: int, path: str) -> str:
    """第 1 段：对每题取 recall_k 个融合候选，缓存下来。"""
    cache = {}
    for item in eval_set:
        cache[item["question"]] = retrieve_fn(item["question"], recall_k)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(cache, f)
    return str(p)


def load_candidates(path: str = DEFAULT_CACHE) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_rerank_fn(cache: dict, reranker, rerank_k: int = 5):
    """第 2 段用的 retrieve_fn：只对缓存里的候选做精排。"""
    def _fn(question: str, k: int | None = None):
        return reranker.rerank(question, cache[question], top_k=k or rerank_k)
    return _fn


def _subset(eval_set: list[dict], per_type: int | None, limit: int | None) -> list[dict]:
    if per_type:
        from eval import sampling
        return sampling.stratified_sample(eval_set, per_type)
    if limit:
        return eval_set[:limit]
    return eval_set


def stage_cache(recall_k: int = 20, retriever: str = "hybrid",
                per_type: int | None = None, limit: int | None = None,
                index_dir: str = "data/qa/index",
                cache_path: str = DEFAULT_CACHE) -> str:
    eval_set = _subset(retrieval_metrics.load_eval_set(), per_type, limit)
    print(f"[段1] 粗排缓存: retriever={retriever} recall_k={recall_k} n={len(eval_set)}",
          flush=True)
    retrieve_fn = retrieval_metrics.build_retriever(retriever, index_dir=index_dir)
    path = cache_candidates(eval_set, retrieve_fn, recall_k, cache_path)
    print(f"[段1] 已缓存 → {path}", flush=True)
    return path


def stage_rerank(k: int = 5, rerank_k: int | None = None,
                 per_type: int | None = None, limit: int | None = None,
                 cache_path: str = DEFAULT_CACHE,
                 out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    from rag.reranker import Reranker

    eval_set = _subset(retrieval_metrics.load_eval_set(), per_type, limit)
    cache = load_candidates(cache_path)
    print(f"[段2] 精排: rerank_k={rerank_k or k} n={len(eval_set)} 缓存={cache_path}",
          flush=True)
    fn = make_rerank_fn(cache, Reranker(), rerank_k=rerank_k or k)
    summary = retrieval_metrics.evaluate_retrieval(fn, eval_set, k=k)

    print(f"HitRate@{k} = {summary['hit_rate@k']:.4f}   MRR = {summary['mrr']:.4f}   "
          f"答案占比 = {summary['answer_ratio@k']:.4f}   未检到 = {summary['miss_count']}/{summary['n']}",
          flush=True)
    print("首次命中排名:", summary["first_hit_rank"], flush=True)
    for t, b in summary["per_type"].items():
        print(f"  {t:<11} HitRate={b['hit_rate@k']:.3f}  MRR={b['mrr']:.3f}  n={b['n']}",
              flush=True)

    payload = {kk: vv for kk, vv in summary.items() if kk != "rows"}
    payload.update({"retriever": "hybrid_rerank", "recall_source": cache_path,
                    "cache": cache_path})
    paths = artifacts.save_run(f"retrieval_hybrid_rerank_k{k}", summary["rows"], payload,
                               out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="两段式重排评估（缓解内存/显存限制）")
    p.add_argument("--stage", required=True, choices=["cache", "rerank"])
    p.add_argument("--recall-k", type=int, default=20)
    p.add_argument("--retriever", default="hybrid", choices=["hybrid", "dense", "bm25"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--rerank-k", type=int, default=None)
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--index-dir", default="data/qa/index")
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    if a.stage == "cache":
        stage_cache(recall_k=a.recall_k, retriever=a.retriever, per_type=a.per_type,
                    limit=a.limit, index_dir=a.index_dir, cache_path=a.cache)
    else:
        stage_rerank(k=a.k, rerank_k=a.rerank_k, per_type=a.per_type, limit=a.limit,
                     cache_path=a.cache, out_dir=a.out)
