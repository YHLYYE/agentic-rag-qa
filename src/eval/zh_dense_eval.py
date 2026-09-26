"""中文轨道的 dense / 混合检索评估（T2Ranking 子集，对着真 qrels）。

为什么单独一个脚本：30k 段落用 bge-m3 在 CPU 上向量化约需 80 分钟，
**必须支持断点续跑**（每批存一个分片，重跑时跳过已算好的分片），
否则中途任何中断都会让整轮白跑。

用法：
    python -m eval.zh_dense_eval --corpus subset_q1000.collection.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval import artifacts, zh_retrieval as zr

DEFAULT_DIR = "data/zh/T2Ranking"
DEFAULT_PREFIX = "subset_q1000"
SHARD = 2000          # 每 2000 段存一次分片
RERANK_HEAD = 30      # 只重排候选头部（CrossEncoder 成本高，且头部决定 nDCG@10）


def embed_cached(texts: list[str], cache_dir: str, prefix: str,
                 model_name: str = "BAAI/bge-m3", dim: int = 1024,
                 device: str | None = None,
                 max_seq_length: int = 1024) -> "list[list[float]]":
    """分批向量化并缓存分片；已存在的分片直接复用（断点续跑）。"""
    import numpy as np

    from rag.embeddings import Embedder

    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    embedder = None
    total = len(texts)
    for start in range(0, total, SHARD):
        end = min(start + SHARD, total)
        path = d / f"{prefix}.emb_{start}_{end}.npy"
        shards.append(path)
        if path.exists():
            print(f"  [复用] {start}-{end}", flush=True)
            continue
        if embedder is None:
            print(f"加载 bge-m3 (device={device or 'auto'}, max_seq_length={max_seq_length}) ...",
                  flush=True)
            embedder = Embedder(model_name=model_name, dim=dim, device=device,
                                show_progress=False, max_seq_length=max_seq_length)
        print(f"  [计算] {start}-{end} / {total}", flush=True)
        vecs = np.array(embedder.embed(texts[start:end]), dtype="float32")
        np.save(path, vecs)          # 先落盘再继续，崩了也不丢
    return [np.load(p) for p in shards]


def make_chunks(pids: list[str], texts: list[str]):
    """chunk_id **必须**是数据集里的真实 pid —— 它要和 qrels 对得上。

    回归教训：曾用合成 id `p{i}`，结果所有指标恒为 0.0（ranking 里的 id 永远
    匹配不上 qrels，却不会报错，只会静默给出 0 分）。这类 bug 只能靠「指标不该
    是 0」的常识 + 单测发现。
    """
    from models import Chunk
    return [Chunk(pids[i], texts[i], pids[i], "", 0) for i in range(len(pids))]


def build_dense_retriever(pids: list[str], texts: list[str], matrix_parts, index_path: str,
                          device: str | None = None, max_seq_length: int = 1024):
    import faiss
    import numpy as np

    mat = np.vstack(matrix_parts).astype("float32")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)

    from rag.embeddings import Embedder
    from rag.dense import DenseRetriever

    chunks = make_chunks(pids, texts)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=mat.shape[1], device=device,
                        show_progress=False, max_seq_length=max_seq_length)
    return chunks, DenseRetriever(chunks, embedder, index_path)


def main(directory: str = DEFAULT_DIR, prefix: str = DEFAULT_PREFIX,
         topk: int = 100, out_dir: str = artifacts.DEFAULT_OUT_DIR,
         device: str | None = None, max_seq_length: int = 1024,
         rerank_model: str | None = None, only: str | None = None) -> dict:
    from rag.bm25 import BM25Retriever, zh_tokenize
    from rag.hybrid import merge_and_rerank

    d = Path(directory)
    corpus_path = d / f"{prefix}.collection.tsv"
    corpus = zr.load_corpus(str(corpus_path))
    queries = zr.load_queries(str(d / f"{prefix}.queries.tsv"))
    qrels = zr.load_qrels(str(d / f"{prefix}.qrels.tsv"))

    pids = list(corpus)
    texts = [corpus[p] for p in pids]
    print(f"语料 {len(pids)} 段 / 查询 {len(queries)} / qrels {sum(len(v) for v in qrels.values())} 条",
          flush=True)

    parts = embed_cached(texts, str(d / "emb_cache"), prefix, device=device,
                         max_seq_length=max_seq_length)
    chunks, dense = build_dense_retriever(pids, texts, parts, str(d / f"{prefix}.faiss"),
                                          device=device, max_seq_length=max_seq_length)
    print("dense 索引就绪", flush=True)

    bm25 = BM25Retriever(chunks, tokenize=zh_tokenize)

    def run(name: str, fn) -> dict:
        print(f"检索中: {name} ...", flush=True)
        rankings = {}
        for i, (qid, text) in enumerate(queries.items(), 1):
            rankings[qid] = [x.chunk.chunk_id for x in fn(text, topk)]
            if i % 200 == 0:
                print(f"    {name}: {i}/{len(queries)}", flush=True)
        s = zr.evaluate_rankings(rankings, qrels, ks=(10, 100))
        s.update({"retriever": name, "corpus_size": len(pids), "topk": topk,
                  "dataset": "T2Ranking(dev subset)", "metrics": "real qrels"})
        print(f"  {name}: ndcg@10={s['ndcg@10']:.4f}  recall@100={s['recall@100']:.4f}  "
              f"mrr@10={s['mrr@10']:.4f}", flush=True)
        rows = [{"qid": q, "query": queries[q], "top10": rankings[q][:10],
                 "ndcg@10": zr.ndcg_at_k(rankings[q], qrels[q], 10),
                 "recall@100": zr.recall_at_k(rankings[q], qrels[q], 100)}
                for q in rankings if q in qrels]
        artifacts.save_run(f"zh_{name}", rows, s, out_dir=out_dir)
        return s

    results = {}
    if only != "hybrid_rerank":          # 已有存档的基线可以跳过，省时间
        results["dense"] = run("dense", lambda q, k: dense.retrieve(q, k))
        results["bm25_zh"] = run("bm25_zh", lambda q, k: bm25.retrieve(q, k))
        results["hybrid"] = run("hybrid", lambda q, k: merge_and_rerank(
            [dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k))

    if rerank_model:
        from rag.reranker import Reranker
        print(f"加载重排模型 {rerank_model} (max_length=512, batch=32) ...", flush=True)
        reranker = Reranker(rerank_model, max_length=512)

        def _rerank(q: str, k: int):
            # 粗排：hybrid 取 k 个候选 → 精排：只重排头部 RERANK_HEAD 个
            # （尾部保持原序 —— 这样 Recall@100 与 hybrid 一致，重排的影响只体现在头部）
            cands = merge_and_rerank([dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k)
            head = reranker.rerank(q, cands[:RERANK_HEAD], top_k=RERANK_HEAD, batch_size=32)
            return head + cands[RERANK_HEAD:]

        results["hybrid_rerank"] = run("hybrid_rerank", _rerank)

    print("\n=== 中文轨道检索对比（T2Ranking 子集，真 qrels，n=%d）===" % len(queries))
    for name, s in results.items():
        print(f"  {name:<10} nDCG@10={s['ndcg@10']:.4f}  Recall@100={s['recall@100']:.4f}  "
              f"MRR@10={s['mrr@10']:.4f}")
    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="中文轨道 dense/混合检索评估")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--device", default="cuda",
                   help="cuda / cpu；cuda 不可用时自动回退 cpu")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="embedding 截断长度（bge-m3 默认 8192 会撑爆显存）")
    p.add_argument("--rerank-model", default=None,
                   help="重排模型路径/名称（如 data/models/bge-reranker-v2-m3）；不给则跳过重排")
    p.add_argument("--only", default=None,
                   help="只跑某个变体（如 hybrid_rerank），跳过已有存档的基线以省时间")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    device = a.device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[提示] CUDA 不可用，回退 CPU（会慢很多）", flush=True)
                device = "cpu"
        except Exception:
            device = "cpu"
    main(directory=a.dir, prefix=a.prefix, topk=a.topk, out_dir=a.out,
         device=device, max_seq_length=a.max_seq_length, rerank_model=a.rerank_model,
         only=a.only)
