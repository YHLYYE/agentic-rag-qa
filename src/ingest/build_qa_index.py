"""从 HotpotQA + TriviaQA 构建 RAG 语料（3 类题各 100 题 + Wikipedia 段落索引）。"""
import pickle
from pathlib import Path

import pandas as pd

from ingest.chunk import chunk_text
from ingest.index import build_index
from rag.embeddings import Embedder


def _hotpot_paragraphs(context) -> list[str]:
    """HotpotQA context: {title:[...], sentences:[[...],...]} → 段落列表。"""
    titles = context.get("title", []) if isinstance(context, dict) else []
    sentences = context.get("sentences", []) if isinstance(context, dict) else []
    out = []
    for i, title in enumerate(titles):
        sents = sentences[i] if i < len(sentences) else []
        out.append((title or "") + " " + " ".join(sents))
    return out


def _trivia_paragraphs(row) -> list[str]:
    """TriviaQA: entity_pages.wiki_context + search_results.search_context。"""
    out = []
    ep = row.get("entity_pages")
    if isinstance(ep, dict) and ep.get("wiki_context") is not None:
        for ctx in ep["wiki_context"]:
            if ctx:
                out.append(str(ctx))
    sr = row.get("search_results")
    if isinstance(sr, dict) and sr.get("search_context") is not None:
        for ctx in sr["search_context"]:
            if ctx:
                out.append(str(ctx))
    return out


def main() -> None:
    out = Path("data/qa")
    out.mkdir(parents=True, exist_ok=True)

    # 1. 抽样
    hp = pd.read_parquet("data/raw_qa/hotpotqa_distractor_val_0.parquet")
    comparison = hp[hp["type"] == "comparison"].sample(100, random_state=42)
    bridge = hp[hp["type"] == "bridge"].sample(100, random_state=42)
    tq = pd.read_parquet("data/raw_qa/trivia_rcweb_val_0.parquet")
    factoid = tq.sample(100, random_state=42)

    # 2. 构建评估集 + 收集段落
    eval_set = []
    seen_texts = set()
    paragraphs = []

    def _add(paras):
        for p in paras:
            p = p.strip()
            if p and p[:80] not in seen_texts:
                seen_texts.add(p[:80])
                paragraphs.append(p)

    for _, r in comparison.iterrows():
        _add(_hotpot_paragraphs(r["context"]))
        eval_set.append({"question": r["question"], "ground_truth": str(r["answer"]), "type": "comparison"})
    for _, r in bridge.iterrows():
        _add(_hotpot_paragraphs(r["context"]))
        eval_set.append({"question": r["question"], "ground_truth": str(r["answer"]), "type": "multi-hop"})
    for _, r in factoid.iterrows():
        _add(_trivia_paragraphs(r))
        ans = r["answer"]
        gt = ans.get("value") if isinstance(ans, dict) else str(ans)
        eval_set.append({"question": r["question"], "ground_truth": str(gt), "type": "factoid"})

    # 3. 切块
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.extend(chunk_text(p, source_doc=f"wiki_{i}", section="", page=0, chunk_size=256))
    print(f"段落 {len(paragraphs)} → chunk {len(chunks)} | 评估题 {len(eval_set)}", flush=True)

    with open(out / "eval_set.pkl", "wb") as f:
        pickle.dump(eval_set, f)

    # 4. 建索引
    print("加载 bge-m3 ...", flush=True)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    build_index(chunks, embedder, str(out / "index"))
    print("完成", flush=True)


if __name__ == "__main__":
    main()
