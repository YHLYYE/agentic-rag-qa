"""对 qrels 的检索评估装置（与语料语言无关）。

为什么这件事比"再加一份中文语料"更重要：
英文那套的检索相关性是用「答案字符串是否出现在 chunk 里」这种**代理指标**衡量的，
有已知盲区（例如答案是 yes 的是非题永远测不出来）。T2Ranking 带**真实人工标注的
4 级相关性**，可以直接测检索质量本身，不再依赖代理。

**这个模块按能力命名、不按语言命名**：只要语料带 qrels，换任何语料都能直接用
（当前默认接 T2Ranking 子集；分词器通过 `--tokenizer` 选择，与语料解耦）。

用法：
    python -m eval.qrels_retrieval --tokenizer zh --topk 100
"""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

from eval import artifacts

DEFAULT_DIR = "data/zh/T2Ranking"
DEFAULT_PREFIX = "subset_q1000"


def load_corpus(path: str) -> dict[str, str]:
    """collection 子集: pid \\t text（无表头）"""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, _, text = line.rstrip("\n").partition("\t")
            if pid:
                out[pid] = text
    return out


def load_queries(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                out.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return out


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """4 级相关性的 nDCG@k（gain = 2^rel - 1）。没有相关文档时返回 0。"""
    gains = [2 ** rels.get(pid, 0) - 1 for pid in ranked[:k]]
    ideal = sorted((2 ** r - 1 for r in rels.values() if r > 0), reverse=True)[:k]
    if not ideal or _dcg(ideal) == 0:
        return 0.0
    return _dcg(gains) / _dcg(ideal)


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {pid for pid, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for i, pid in enumerate(ranked[:k], 1):
        if rels.get(pid, 0) > 0:
            return 1.0 / i
    return 0.0


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]],
                      ks: tuple[int, ...] = (10, 100)) -> dict:
    """rankings: {qid: [pid...]}（按相关度降序）。只统计 qrels 里出现了的 query。"""
    usable = [q for q in rankings if q in qrels]
    summary: dict = {"n": len(usable)}
    for k in ks:
        summary[f"ndcg@{k}"] = sum(ndcg_at_k(rankings[q], qrels[q], k) for q in usable) / len(usable)
        summary[f"recall@{k}"] = sum(recall_at_k(rankings[q], qrels[q], k) for q in usable) / len(usable)
        summary[f"mrr@{k}"] = sum(mrr_at_k(rankings[q], qrels[q], k) for q in usable) / len(usable)
    return summary


def build_bm25_tokenizer(kind: str):
    from rag.bm25 import simple_tokenize, tokenize_no_stopwords, zh_tokenize
    if kind == "zh":
        return zh_tokenize
    if kind == "en":
        return tokenize_no_stopwords
    if kind == "en_split":
        return simple_tokenize
    raise ValueError(f"unknown tokenizer: {kind}")


def main(directory: str = DEFAULT_DIR, prefix: str = DEFAULT_PREFIX,
         tokenizer: str = "zh", topk: int = 100,
         out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    from models import Chunk
    from rag.bm25 import BM25Retriever

    d = Path(directory)
    corpus = load_corpus(str(d / f"{prefix}.collection.tsv"))
    queries = load_queries(str(d / f"{prefix}.queries.tsv"))
    qrels = load_qrels(str(d / f"{prefix}.qrels.tsv"))
    print(f"语料 {len(corpus)} 段 / 查询 {len(queries)} / qrels {sum(len(v) for v in qrels.values())} 条",
          flush=True)

    chunks = [Chunk(pid, text, pid, "", 0) for pid, text in corpus.items()]
    tok = build_bm25_tokenizer(tokenizer)
    print(f"构建 BM25 索引（tokenizer={tokenizer}, {len(chunks)} 段）...", flush=True)
    bm25 = BM25Retriever(chunks, tokenize=tok)

    print(f"检索 top-{topk} ...", flush=True)
    rankings = {qid: [x.chunk.chunk_id for x in bm25.retrieve(text, topk)]
                for qid, text in queries.items()}

    summary = evaluate_rankings(rankings, qrels, ks=(10, 100))
    summary.update({"tokenizer": tokenizer, "corpus_size": len(corpus),
                    "topk": topk, "dataset": "T2Ranking(dev subset)"})
    print("\n=== 中文检索评估（T2Ranking 子集，真 qrels）===", flush=True)
    print(f"ndcg@10 = {summary['ndcg@10']:.4f}   recall@100 = {summary['recall@100']:.4f}   "
          f"mrr@10 = {summary['mrr@10']:.4f}", flush=True)

    rows = [{"qid": q, "query": queries[q],
             "top10": rankings[q][:10],
             "ndcg@10": ndcg_at_k(rankings[q], qrels[q], 10),
             "recall@100": recall_at_k(rankings[q], qrels[q], 100)}
            for q in rankings if q in qrels]
    # run 标签统一为能力命名；改名前产生的存档仍保留 zh_retrieval_* 前缀（历史记录不改写）
    paths = artifacts.save_run(f"qrels_retrieval_{tokenizer}", rows, summary, out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="中文检索评估（T2Ranking 子集，真 qrels）")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--tokenizer", default="zh", choices=["zh", "en", "en_split"])
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    main(directory=a.dir, prefix=a.prefix, tokenizer=a.tokenizer, topk=a.topk, out_dir=a.out)
