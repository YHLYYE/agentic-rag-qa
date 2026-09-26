"""从 T2Ranking 构建**中文检索评估子集**（原生中文 + 真实人工相关性标注）。

为什么需要它：T2Ranking 的 collection 有 3.4GB / 300 万+ 段落，本机（CPU、可用内存
~2GB）无法建稠密索引。所以下采样，但要下采样得**方法论上站得住**：

    1. 选 N 条查询（只选「至少有 1 条正例」的，否则该题算不出召回）
    2. 收进这 N 条查询**全部被判定过**的段落（正例 + 负例）
       —— 保证评价这些查询时 qrels 在子集内是**完整的**，不会因为缺段落而虚高/虚低
    3. 再从判定池里随机补一批干扰段落（让任务不至于「只在少量候选里挑」）

代价必须说清楚：这是在 **pooled 判定池**上评测（标准 IR 做法），比在 300 万段落全量上
找要**容易**。引用指标时要带上「语料规模」这个前提。

用法：
    python -m ingest.build_zh_t2ranking --queries 1000 --distractors 14000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_DIR = "data/zh/T2Ranking"


def load_queries(path: str) -> dict[str, str]:
    """queries.dev.tsv: qid \\t text（首行表头）"""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                out[parts[0]] = parts[1]
    return out


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """qrels.dev.tsv: qid \\t 0 \\t pid \\t rel（首行表头）；返回 {qid: {pid: rel}}"""
    out: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                out.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return out


def select_queries(qrels: dict[str, dict[str, int]], n: int,
                   seed: int = 42) -> list[str]:
    """只选至少有 1 条正例的查询（否则召回恒为 0，是噪声不是信号）。结果确定性。"""
    usable = sorted(q for q, rels in qrels.items() if any(r > 0 for r in rels.values()))
    if n >= len(usable):
        return usable
    return sorted(random.Random(seed).sample(usable, n))


def select_pids(qrels: dict[str, dict[str, int]], qids: list[str],
                n_distractors: int, seed: int = 42) -> tuple[set[str], set[str]]:
    """返回 (本批查询判定过的全部 pid, 随机干扰 pid)。两组不重叠。"""
    judged: set[str] = set()
    for q in qids:
        judged.update(qrels.get(q, {}))
    pool = sorted({p for rels in qrels.values() for p in rels} - judged)
    extra = set(pool) if n_distractors >= len(pool) else set(
        random.Random(seed).sample(pool, n_distractors))
    return judged, extra


def extract_subset(collection_path: str, pids: set[str], out_path: str) -> int:
    """流式扫描 collection.tsv（pid \\t text），只写出目标 pid。3.4GB 也不会吃内存。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(collection_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            pid, _, text = line.partition("\t")
            if pid in pids:
                fout.write(line)
                written += 1
    return written


def main(directory: str = DEFAULT_DIR, queries: int = 1000,
         distractors: int = 14000, seed: int = 42,
         out_prefix: str | None = None) -> dict:
    d = Path(directory)
    prefix = out_prefix or f"subset_q{queries}"
    qpath, rpath = d / "queries.dev.tsv", d / "qrels.dev.tsv"

    all_queries = load_queries(str(qpath))
    qrels = load_qrels(str(rpath))
    qids = select_queries(qrels, queries, seed=seed)
    judged, extra = select_pids(qrels, qids, distractors, seed=seed)
    keep = judged | extra

    print(f"候选查询 {len(all_queries)} → 选中 {len(qids)}（均有正例）", flush=True)
    print(f"判定段落 {len(judged)} + 干扰段落 {len(extra)} = 目标 {len(keep)}", flush=True)
    written = extract_subset(str(d / "collection.tsv"), keep,
                             str(d / f"{prefix}.collection.tsv"))
    print(f"实际写出段落 {written}（缺失 {len(keep) - written}，多为 collection 未收录）",
          flush=True)

    # 落盘这一批查询与它们的 qrels（只保留子集内的段落，保证一致）
    with open(d / f"{prefix}.queries.tsv", "w", encoding="utf-8") as f:
        f.write("qid\ttext\n")
        for q in qids:
            f.write(f"{q}\t{all_queries.get(q, '')}\n")
    n_rel = 0
    with open(d / f"{prefix}.qrels.tsv", "w", encoding="utf-8") as f:
        f.write("qid\t0\tpid\trel\n")
        for q in qids:
            for pid, rel in sorted(qrels.get(q, {}).items()):
                if pid in keep:
                    f.write(f"{q}\t0\t{pid}\t{rel}\n")
                    n_rel += 1
    print(f"已落盘: {prefix}.collection.tsv / {prefix}.queries.tsv / {prefix}.qrels.tsv"
          f"（qrels 保留 {n_rel} 条）", flush=True)
    return {"queries": len(qids), "passages": written, "qrels": n_rel}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="构建 T2Ranking 中文检索评估子集")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--distractors", type=int, default=14000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    main(directory=a.dir, queries=a.queries, distractors=a.distractors,
         seed=a.seed, out_prefix=a.out_prefix)
