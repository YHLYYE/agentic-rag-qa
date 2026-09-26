"""qrels 语料的**生成侧与工程数据**：忠实度（faithfulness）+ 检索延迟基准。

为什么需要这个模块：`qrels_retrieval` / `qrels_dense_eval` 只覆盖**检索侧**（那套语料
没有标准答案，所以测不了答案正确率）。但有两件事**不依赖标准答案**，因此可以补：
1. **忠实度**：判「答案的每个断言是否被检索到的上下文支持」—— 不需要 gold answer
2. **延迟**：工程数据，和语料无关

注意：延迟要**标明设备**。英文轨道那组是 CPU（BM25 21ms / dense 124ms），本模块走 GPU，
两者**不可直接比较**。

用法：
    python -m eval.qrels_gen_latency --faithfulness 30 --latency 30 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval import artifacts, qrels_retrieval as zr, sampling
from eval.ragas_self import faithfulness
from eval.retrieval_metrics import bench_latency

DEFAULT_DIR = "data/zh/T2Ranking"
DEFAULT_PREFIX = "subset_q1000"
GEN_PROMPT = "根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}"


def build_all(directory: str, prefix: str, device: str | None,
              rerank_model: str | None):
    """返回 {名字: retrieve_fn}；dense 直接加载已导出的索引（不重新向量化）。"""
    import pickle

    from rag.bm25 import BM25Retriever, zh_tokenize
    from rag.dense import DenseRetriever
    from rag.embeddings import Embedder
    from rag.hybrid import merge_and_rerank

    d = Path(directory)
    index_dir = d / "index"
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)          # 复用 export_zh_index 的产物
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024, device=device,
                        show_progress=False)
    dense = DenseRetriever(chunks, embedder, str(index_dir / "faiss.index"))
    bm25 = BM25Retriever(chunks, tokenize=zh_tokenize)

    fns = {
        "bm25": lambda q, k: bm25.retrieve(q, k),
        "dense": lambda q, k: dense.retrieve(q, k),
        "hybrid": lambda q, k: merge_and_rerank(
            [dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k),
    }
    if rerank_model:
        from rag.reranker import Reranker
        reranker = Reranker(rerank_model, max_length=512)

        def _rr(q: str, k: int):
            cands = merge_and_rerank([dense.retrieve(q, k), bm25.retrieve(q, k)], top_k=k)
            head = reranker.rerank(q, cands[:30], top_k=30, batch_size=32)
            return head + cands[30:]

        fns["hybrid_rerank"] = _rr
    return fns


def main(directory: str = DEFAULT_DIR, prefix: str = DEFAULT_PREFIX,
         faithfulness_n: int = 30, latency_n: int = 30, topk: int = 5,
         device: str | None = "cuda",
         rerank_model: str | None = "data/models/bge-reranker-v2-m3",
         out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    queries = zr.load_queries(str(Path(directory) / f"{prefix}.queries.tsv"))
    # 确定性抽样：用现成的分层抽样工具（这里没有题型，等价于按固定种子取样）
    items = [{"type": "all", "id": qid, "text": text} for qid, text in queries.items()]
    picked = sampling.stratified_sample(items, faithfulness_n) if faithfulness_n else []

    print("加载检索器与模型 ...", flush=True)
    fns = build_all(directory, prefix, device, rerank_model)

    result: dict = {"n_queries": len(queries), "device": device,
                    "available_retrievers": sorted(fns)}

    # ---- 1. 延迟基准（工程数据）----
    if latency_n:
        lat_questions = [it["text"] for it in items[:latency_n]]
        print(f"\n=== 延迟基准（{latency_n} 条查询, k={topk}, device={device}）===", flush=True)
        lat = {}
        for name, fn in fns.items():
            s = bench_latency(fn, lat_questions, k=topk)
            lat[name] = s
            print(f"  {name:<14} p50={s['p50_ms']:7.1f}ms  p95={s['p95_ms']:7.1f}ms  "
                  f"mean={s['mean_ms']:7.1f}ms", flush=True)
        result["latency_ms"] = lat

    # ---- 2. 忠实度（不需要标准答案）----
    if faithfulness_n:
        import os

        from dotenv import load_dotenv
        from openai import OpenAI

        from llm import DeepSeekLLM

        load_dotenv()
        client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"],
                        api_key=os.environ["DEEPSEEK_API_KEY"])
        llm = DeepSeekLLM(client)
        fn = fns.get("hybrid_rerank") or fns["hybrid"]

        print(f"\n=== 忠实度（{len(picked)} 条查询, 检索器=hybrid_rerank）===", flush=True)
        rows, scores = [], []
        for i, it in enumerate(picked, 1):
            q = it["text"]
            rc = fn(q, topk)
            ctx = "\n".join(x.chunk.text for x in rc)
            ans = llm.complete(GEN_PROMPT.format(q=q, ctx=ctx))
            f = faithfulness(client, q, ans, ctx)
            scores.append(f)
            rows.append({"qid": it["id"], "query": q, "answer": ans,
                         "retrieved_chunk_ids": [x.chunk.chunk_id for x in rc],
                         "faithfulness": f})
            print(f"  [{i}/{len(picked)}] f={f:.2f}  {q[:34]}", flush=True)

        result["faithfulness"] = sum(scores) / len(scores)
        result["faithfulness_n"] = len(scores)
        print(f"\n忠实度 = {result['faithfulness']:.4f}  (n={len(scores)}, "
              f"注意：该语料无标准答案，因此不报答案正确率)", flush=True)
        payload = {k: v for k, v in result.items()}
        payload["dataset"] = "T2Ranking(dev subset)"
        paths = artifacts.save_run("qrels_gen_latency", rows, payload, out_dir=out_dir)
        print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="qrels 语料的忠实度 + 延迟基准")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--faithfulness", type=int, default=30, help="忠实度评测的查询数")
    p.add_argument("--latency", type=int, default=30, help="延迟基准的查询数")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rerank-model", default="data/models/bge-reranker-v2-m3")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    main(directory=a.dir, prefix=a.prefix, faithfulness_n=a.faithfulness,
         latency_n=a.latency, topk=a.topk, device=a.device,
         rerank_model=a.rerank_model, out_dir=a.out)
