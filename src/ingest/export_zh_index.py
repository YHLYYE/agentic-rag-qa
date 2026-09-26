"""把 T2Ranking 子集导成和英文轨道**同构**的索引产物（chunks.pkl + faiss.index）。

为什么要这一步：交互式入口（run_graph / Streamlit）统一按「index_dir 下有
chunks.pkl + faiss.index」来加载语料，而中文语料原本是 TSV + 一个不同命名的 faiss。
导出成同构产物之后，中文语料就能直接被问答入口使用。

embedding 复用 `qrels_dense_eval` 的分片缓存（`emb_cache/`），所以不会重新向量化。

用法：
    python -m ingest.export_zh_index            # 需要先跑过 qrels_dense_eval（有缓存分片）
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np

from eval import qrels_retrieval as zr

DEFAULT_DIR = "data/zh/T2Ranking"
DEFAULT_PREFIX = "subset_q1000"


def export(directory: str = DEFAULT_DIR, prefix: str = DEFAULT_PREFIX,
           out_dir: str | None = None) -> dict:
    from models import Chunk

    d = Path(directory)
    out = Path(out_dir) if out_dir else d / "index"
    out.mkdir(parents=True, exist_ok=True)

    corpus = zr.load_corpus(str(d / f"{prefix}.collection.tsv"))
    pids = list(corpus)
    texts = [corpus[p] for p in pids]

    shards = sorted((d / "emb_cache").glob(f"{prefix}.emb_*.npy"))
    if not shards:
        raise SystemExit(
            f"找不到 embedding 分片（{(d / 'emb_cache')}）。\n"
            f"请先跑：python -m qrels_dense_eval  或从 qrels_dense_eval 侧重建缓存")
    mat = np.vstack([np.load(p) for p in shards]).astype("float32")
    if mat.shape[0] != len(pids):
        raise SystemExit(f"分片行数 {mat.shape[0]} 与语料 {len(pids)} 不一致，缓存可能不完整")

    # chunk_id 必须是真实 pid（要和 qrels 对得上；这条踩过坑）
    chunks = [Chunk(pids[i], texts[i], pids[i], "", 0) for i in range(len(pids))]
    with open(out / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, str(out / "faiss.index"))

    print(f"导出完成: {out}/chunks.pkl ({len(chunks)} 段) + faiss.index (dim={mat.shape[1]})",
          flush=True)
    return {"chunks": len(chunks), "dim": int(mat.shape[1]), "out_dir": str(out)}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="导出中文语料的统一索引产物")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    export(directory=a.dir, prefix=a.prefix, out_dir=a.out_dir)
