"""基准守护 —— 对账「公开文档里引用的数字」和「存档 + 逐题明细」。

要解决的问题：数字和代码/存档脱钩。这个仓库已经踩过三次：
  · 存档记的 git_rev 在历史重写后失效
  · 改写保真率两次跑出 0.2 / 1.0，却按 1.0 对外引用
  · README 写着 token 降 74.8%，而当前代码跑出来是 63.7%

对每条「对外引用过的数字」，做三道检查：

  ① 自洽   —— 从逐题明细重算，和存档里的汇总对得上吗？
              （对不上 = 汇总可能是手写的，不是算出来的）
  ② 取值   —— 文档里写的四舍五入值，和存档里的精确值一致吗？
              （同一份值在不同文档可以精度不同：README 写 0.9667、简历写 0.97）
  ③ 有出处 —— 这个字符串确实出现在那份文档里吗？

三道全过才算 PASS，任何一条不过 → 退出码非 0。

只用标准库：不加载 faiss / 模型 / API，任何机器、任何环境都能跑。

用法：
    python -m eval.verify_claims
    python -m eval.verify_claims --resume "C:/path/to/简历.md"
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNS_DIR = Path("data/reports/runs")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 数字依赖的派生文件（被 .gitignore 排除）。缺了不代表数字错，
# 只代表「在别的机器上不能一键重跑」——列出来是为了别在面试现场翻车。
PREREQUISITES = [
    ("data/zh/T2Ranking/index/faiss.index", "中文轨道向量索引"),
    ("data/zh/T2Ranking/index/chunks.pkl", "中文轨道 chunk 存储"),
    ("data/zh/T2Ranking/emb_cache", "中文轨道 embedding 分片"),
    ("data/models", "bge-m3 / bge-reranker 权重"),
    ("data/qa/index/faiss.index", "QA 语料向量索引"),
]


@dataclass
class Claim:
    """一条对外声明过的数字。

    quoted: {文档路径: 该文档里写的字符串}。同一份值在不同文档可以精度不同，
    脚本会逐个核对「文档写的字符串 == 存档值按同样精度格式化」。
    """
    label: str
    archive: str                      # data/reports/runs/ 下的存档文件名
    field: str                        # 要核的字段名
    recompute: tuple                  # ("mean_num"|"mean_bool"|"mrr", 明细字段名)
    quoted: dict[str, str] = field(default_factory=dict)
    nested: str | None = None         # 字段在 summary 的子对象里时填子对象名
    # 已经被撤下的旧值：文档里只要还能搜到就算失败。
    # 这几个都是真实踩过的坑——数字撤了，文档里的残留没清干净。
    stale: list[str] = field(default_factory=list)


DEFAULT_RESUME = "../简历-2026秋招.md"
T2_DOC = "data/reports/中文检索轨道-T2Ranking.md"

CLAIMS: list[Claim] = [
    # ── 中文检索轨道：BM25（中文分词，主对照）──
    Claim("中文 BM25（中文分词）nDCG@10",
          "20260926T114336Z_zh_bm25_zh.summary.json", "ndcg@10",
          ("mean_num", "ndcg@10"), {"README.md": "0.4749"}),
    Claim("中文 BM25（中文分词）Recall@100",
          "20260926T114336Z_zh_bm25_zh.summary.json", "recall@100",
          ("mean_num", "recall@100"), {"README.md": "0.8390"}),
    Claim("中文 BM25（中文分词）MRR@10",
          "20260926T114336Z_zh_bm25_zh.summary.json", "mrr@10",
          ("mean_num", "mrr@10"), {"README.md": "0.7116"}),

    # ── 中文检索轨道：英文分词器作为对照（证明瓶颈在分词）──
    Claim("中文 BM25（英文分词器，对照）nDCG@10",
          "20260926T100329Z_zh_retrieval_en_split.summary.json", "ndcg@10",
          ("mean_num", "ndcg@10"), {"README.md": "0.0498"}),
    Claim("中文 BM25（英文分词器，对照）Recall@100",
          "20260926T100329Z_zh_retrieval_en_split.summary.json", "recall@100",
          ("mean_num", "recall@100"), {"README.md": "0.0941"}),
    Claim("中文 BM25（英文分词器，对照）MRR@10",
          "20260926T100329Z_zh_retrieval_en_split.summary.json", "mrr@10",
          ("mean_num", "mrr@10"), {"README.md": "0.0763"}),

    # ── 中文检索轨道：混合 + 统一重排（最优）──
    Claim("中文 hybrid+统一重排 nDCG@10",
          "20260926T151111Z_zh_hybrid_rerank.summary.json", "ndcg@10",
          ("mean_num", "ndcg@10"),
          {"README.md": "0.6021", T2_DOC: "0.6021", DEFAULT_RESUME: "0.6021"}),
    Claim("中文 hybrid+统一重排 Recall@100",
          "20260926T151111Z_zh_hybrid_rerank.summary.json", "recall@100",
          ("mean_num", "recall@100"), {T2_DOC: "0.9543"}),

    # ── 英文对照轮（n=30 分层抽样）──
    Claim("路由准确率（n=30）",
          "20260925T172221Z_route_accuracy.summary.json", "accuracy",
          ("mean_bool", "correct"), {"README.md": "0.833"},
          stale=["84.3%"]),
    Claim("QA 轮 faithfulness（n=30）",
          "20260925T172641Z_qa_ragas.summary.json", "faithfulness",
          ("mean_num", "faithfulness"), {"README.md": "0.856"}, nested="overall"),
    Claim("QA 轮 context_precision（n=30）",
          "20260925T172641Z_qa_ragas.summary.json", "context_precision",
          ("mean_num", "context_precision"), {"README.md": "0.207"}, nested="overall"),
    Claim("QA 轮 context_recall（n=30）",
          "20260925T172641Z_qa_ragas.summary.json", "context_recall",
          ("mean_num", "context_recall"), {"README.md": "0.833"}, nested="overall"),

    # ── 端到端答案正确率（简历里写的是两位小数）──
    Claim("端到端答案正确率（hybrid_rerank）",
          "20260925T193001Z_answer_eval_hybrid_rerank.summary.json", "answer_correctness",
          ("mean_num", "answer_correctness"), {DEFAULT_RESUME: "0.97"}),
    Claim("端到端答案正确率（dense 对照）",
          "20260925T184925Z_answer_eval_dense.summary.json", "answer_correctness",
          ("mean_num", "answer_correctness"), {DEFAULT_RESUME: "0.90"}),

    # ── 多轮改写消融 ──
    Claim("改写消融：不改写 HitRate@5",
          "20260926T165836Z_rewrite_ablation.summary.json", "hit_follow_up(不改写)",
          ("mean_bool", "hit_follow_up"),
          {"README.md": "0.15", DEFAULT_RESUME: "0.15"}),
    Claim("改写消融：改写后 HitRate@5",
          "20260926T165836Z_rewrite_ablation.summary.json", "hit_rewritten(实际方案)",
          ("mean_bool", "hit_rewritten"),
          {"README.md": "0.95", DEFAULT_RESUME: "0.95"},
          stale=["20/20", "保真率 20"]),
]


def _rows(detail: Path) -> list[dict]:
    return [json.loads(l) for l in detail.read_text(encoding="utf-8").splitlines() if l.strip()]


def _recompute(rows: list[dict], spec: tuple) -> float | None:
    kind, key = spec
    if kind == "mean_bool":
        vals = [bool(r.get(key)) for r in rows if r.get(key) is not None]
    elif kind == "mean_num":
        vals = [r.get(key) for r in rows if r.get(key) is not None]
    elif kind == "mrr":
        vals = [(1.0 / r[key]) if r.get(key) else 0.0 for r in rows]
    else:
        return None
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float, like: str) -> str:
    """按文档里那个字符串的小数位，把存档值格式化成同样的精度。"""
    decimals = len(like.split(".")[1]) if "." in like else 0
    return f"{value:.{decimals}f}"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _check_claim(c: Claim) -> tuple[bool, list[str], float | None]:
    notes: list[str] = []
    ok = True

    archive_path = RUNS_DIR / c.archive
    if not archive_path.exists():
        return False, [f"存档不存在：{c.archive}"], None
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    scope = archive.get("summary", {})
    if c.nested:
        scope = scope.get(c.nested, {})
    stored = scope.get(c.field)
    if stored is None:
        return False, [f"存档里没有字段 {c.nested + '.' if c.nested else ''}{c.field}"], None

    # ① 自洽：逐题明细重算 == 存档汇总
    detail_name = archive.get("detail_file")
    detail_path = RUNS_DIR / detail_name if detail_name else None
    if detail_path and detail_path.exists():
        recalculated = _recompute(_rows(detail_path), c.recompute)
        if recalculated is None:
            notes.append(f"明细里没有 {c.recompute[1]} 字段，自洽检查跳过（该指标只能靠存档）")
        elif abs(recalculated - stored) > 1e-9:
            ok = False
            notes.append(f"自洽失败：明细重算 {recalculated:.10f} != 存档 {stored:.10f}")
    else:
        notes.append("该存档没有逐题明细，自洽检查跳过")

    # ②③ 每个引用它的文档：精度对齐 + 字符串确实在
    for doc, shown in c.quoted.items():
        expected = _fmt(stored, shown)
        if expected != shown:
            ok = False
            notes.append(f"取值不符：{doc} 写 {shown}，存档按同精度应为 {expected}")
        doc_path = _resolve(doc)
        if not doc_path.exists():
            notes.append(f"文档不存在，跳过引用检查：{doc}")
            continue
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        if shown not in text:
            ok = False
            notes.append(f"{doc} 里找不到字符串 {shown}")
        for s in c.stale:
            if s in text:
                ok = False
                notes.append(f"{doc} 里残留旧值 {s}")

    return ok, notes, stored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="对账文档引用的数字与存档")
    ap.add_argument("--resume", default=DEFAULT_RESUME,
                    help=f"简历路径（默认 {DEFAULT_RESUME}；不存在则跳过相关声明）")
    args = ap.parse_args(argv)

    total = passed = 0
    failures: list[str] = []

    print("=" * 78)
    print(f"基准守护 — 核对 {len(CLAIMS)} 条对外声明")
    print("=" * 78)

    for c in CLAIMS:
        ok, notes, stored = _check_claim(c)
        total += 1
        passed += 1 if ok else 0
        if not ok:
            failures.append(c.label)
        print(f"\n{'✅' if ok else '❌'} {c.label}")
        print(f"     存档 {c.archive}")
        if stored is not None:
            print(f"     精确值 {stored:.10f}")
        for doc, shown in c.quoted.items():
            print(f"     引用 {doc} -> {shown}")
        if c.stale:
            print(f"     不得残留 {'、'.join(c.stale)}")
        for n in notes:
            print(f"     · {n}")

    print("\n" + "=" * 78)
    print("复现前提（派生文件，被 .gitignore 排除）")
    print("=" * 78)
    missing = []
    for path, label in PREREQUISITES:
        exists = _resolve(path).exists()
        print(f"{'✅' if exists else '⬜'} {label:<26} {path}")
        if not exists:
            missing.append(label)
    if missing:
        print(f"\n⬜ 缺 {len(missing)} 项 —— 数字本身没问题，但换台机器需先重建才能重跑")

    print("\n" + "=" * 78)
    if failures:
        print(f"结果：❌ {total - passed}/{total} 条不通过")
        for f in failures:
            print(f"   · {f}")
        return 1
    print(f"结果：✅ 全部 {total} 条通过（自洽 + 取值 + 有出处）")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    raise SystemExit(main())
