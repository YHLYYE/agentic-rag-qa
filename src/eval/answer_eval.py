"""端到端评估：答案正确率 + 召回/忠实度，一次跑齐（检索 → 生成 → 判官）。

这是本项目**唯一**能回答「系统到底答得对不对」的脚本。此前所有指标都只测
「答案有没有被检索到」（HitRate）或「答案信息点在不在上下文里」（context_recall），
都不是「最终答案是否正确」。

内存约束：本机无法在同一进程里同时装 bge-m3 和 bge-reranker，所以
`--retriever hybrid_rerank` 复用第 1 段缓存好的候选（`--cache`），此进程只加载 reranker。

用法：
    python -m eval.answer_eval --retriever hybrid_rerank --cache data/qa/hybrid_candidates_k50.pkl --per-type 10
    python -m eval.answer_eval --retriever dense --per-type 10        # 对照：旧配置
"""
from __future__ import annotations

import argparse
import collections
import statistics

from eval import artifacts, sampling
from eval.ragas_self import (answer_correctness, context_precision, context_recall,
                             faithfulness)
from eval.retrieval_metrics import build_retriever, load_eval_set

GEN_PROMPT = "根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}"

METRIC_KEYS = ("faithfulness", "context_precision", "context_recall")


def summarize(rows: list[dict]) -> dict:
    """汇总时把 answer_correctness 为 None（判官解析失败）的样本排除出分母，并单独计数。"""
    scored = [r["answer_correctness"] for r in rows if r["answer_correctness"] is not None]

    def _block(sub: list[dict]) -> dict:
        ok = [r["answer_correctness"] for r in sub if r["answer_correctness"] is not None]
        block = {"n": len(sub),
                 "answer_correctness": (sum(ok) / len(ok)) if ok else None,
                 "answer_correctness_parse_failures": len(sub) - len(ok)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in sub if r.get(k) is not None]
            block[k] = statistics.fmean(vals) if vals else None
        return block

    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    return {
        **_block(rows),
        "per_type": {t: _block(v) for t, v in sorted(by_type.items())},
    }


def _make_retriever(kind: str, cache: str, index_dir: str):
    if kind == "hybrid_rerank":
        from eval.rerank_eval import load_candidates, make_rerank_fn
        from rag.reranker import Reranker
        # 只加载 reranker（不碰 bge-m3）——候选来自第 1 段缓存
        return make_rerank_fn(load_candidates(cache), Reranker())
    return build_retriever(kind, index_dir=index_dir)


def run(retriever: str = "hybrid_rerank", cache: str = "data/qa/hybrid_candidates_k50.pkl",
        per_type: int | None = 10, limit: int | None = None, k: int = 5,
        index_dir: str = "data/qa/index",
        out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    from llm import DeepSeekLLM

    load_dotenv()
    client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"],
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    llm = DeepSeekLLM(client)

    eval_set = load_eval_set()
    if per_type:
        eval_set = sampling.stratified_sample(eval_set, per_type)
    elif limit:
        eval_set = eval_set[:limit]

    print(f"端到端评估: retriever={retriever} k={k} n={len(eval_set)}", flush=True)
    retrieve_fn = _make_retriever(retriever, cache, index_dir)

    rows = []
    for i, item in enumerate(eval_set, 1):
        q, gt, t = item["question"], item["ground_truth"], item["type"]
        rc = retrieve_fn(q, k)
        ctx = "\n".join(x.chunk.text for x in rc)
        ans = llm.complete(GEN_PROMPT.format(q=q, ctx=ctx))
        ac = answer_correctness(client, q, ans, gt)
        row = {
            "question": q, "type": t, "ground_truth": gt, "answer": ans,
            "retrieved_chunk_ids": [x.chunk.chunk_id for x in rc],
            "answer_correctness": ac["score"],
            "answer_correctness_raw": ac["raw"],      # 判官原文，支持事后审计
            "faithfulness": faithfulness(client, q, ans, ctx),
            "context_precision": context_precision(client, q, ctx),
            "context_recall": context_recall(client, q, ctx, gt),
        }
        rows.append(row)
        print(f"[{i}/{len(eval_set)}] {t:<11} 正确={ac['score']} "
              f"f={row['faithfulness']:.2f} r={row['context_recall']:.2f}", flush=True)

    summary = summarize(rows)
    summary["retriever"] = retriever
    summary["cache"] = cache if retriever == "hybrid_rerank" else None
    print("\n=== 端到端评估结果 ===", flush=True)
    ac = summary["answer_correctness"]
    print(f"答案正确率 = {ac if ac is None else round(ac, 4)}  "
          f"(判官解析失败 {summary['answer_correctness_parse_failures']}/{summary['n']})", flush=True)
    for key in METRIC_KEYS:
        print(f"  {key} = {round(summary[key], 4)}", flush=True)
    for t, b in summary["per_type"].items():
        print(f"  {t:<11} 正确率={b['answer_correctness']}  "
              f"f={round(b['faithfulness'],3)} r={round(b['context_recall'],3)}", flush=True)

    paths = artifacts.save_run(f"answer_eval_{retriever}", rows, summary, out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="端到端评估：答案正确率 / 召回 / 忠实度")
    p.add_argument("--retriever", default="hybrid_rerank",
                   choices=["hybrid_rerank", "dense", "bm25", "hybrid"])
    p.add_argument("--cache", default="data/qa/hybrid_candidates_k50.pkl")
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index-dir", default="data/qa/index")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    run(retriever=a.retriever, cache=a.cache, per_type=a.per_type, limit=a.limit,
        k=a.k, index_dir=a.index_dir, out_dir=a.out)
