"""对 HotpotQA + TriviaQA 语料跑 RAGAS（faithfulness / precision / recall），分题型统计并逐题落盘。"""
import argparse
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

from eval import artifacts, sampling
from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank
from graph.nodes import route_node
from eval.ragas_self import faithfulness, context_precision, context_recall


class LLM:
    def __init__(self, client: OpenAI):
        self.client = client

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content.strip()


def main(n: int = 300, per_type: int | None = None,
         out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    load_dotenv()
    client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"], api_key=os.environ["DEEPSEEK_API_KEY"])
    llm = LLM(client)

    with open("data/qa/eval_set.pkl", "rb") as f:
        all_items = pickle.load(f)
    if per_type:
        eval_set = sampling.stratified_sample(all_items, per_type)
        sampling_desc = {"mode": "stratified", "per_type": per_type}
    else:
        eval_set = all_items[:n]
        sampling_desc = {"mode": "prefix", "limit": n}

    covered = {i["type"] for i in eval_set}
    if len(covered) == 1:
        print(f"[警告] 抽样只覆盖单一题型 {covered}，结论不可外推；建议改用 --per-type",
              flush=True)
    with open("data/qa/index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    # device="cpu"：本机 8GB 显存跑 bge-m3 会 OOM（与 Streamlit 同样的处置）
    # show_progress=False：进度条写 stderr，会污染日志与存档输出
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024, device="cpu",
                        show_progress=False)
    dense = DenseRetriever(chunks, embedder, "data/qa/index/faiss.index")
    bm25 = BM25Retriever(chunks)

    results = {t: {"f": [], "p": [], "r": []} for t in ("factoid", "comparison", "multi-hop")}
    rows = []
    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        strategy = route_node({"question": q}, llm=llm)["intent"]
        if strategy == "keyword":
            rc = bm25.retrieve(q, 5)
        elif strategy == "hybrid":
            rc = merge_and_rerank([dense.retrieve(q, 10), bm25.retrieve(q, 10)], top_k=5)
        else:
            rc = dense.retrieve(q, 5)
        ctx = "\n".join(x.chunk.text for x in rc)
        ans = llm.complete(f"根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}")
        t = item["type"]
        f = faithfulness(client, q, ans, ctx)
        p = context_precision(client, q, ctx)
        r = context_recall(client, q, ctx, item["ground_truth"])
        results[t]["f"].append(f)
        results[t]["p"].append(p)
        results[t]["r"].append(r)
        rows.append({
            "question": q,
            "type": t,
            "strategy": strategy,
            "retrieved_chunk_ids": [x.chunk.chunk_id for x in rc],
            "answer": ans,
            "faithfulness": f,
            "context_precision": p,
            "context_recall": r,
        })
        print(f"[{i}/{len(eval_set)}] {t} f={f:.2f} p={p:.2f} r={r:.2f}", flush=True)

    print("=== RAGAS 分题型 ===", flush=True)
    all_f, all_p, all_r = [], [], []
    per_type = {}
    for t, m in results.items():
        if m["f"]:
            f = sum(m["f"]) / len(m["f"])
            p = sum(m["p"]) / len(m["p"])
            r = sum(m["r"]) / len(m["r"])
            all_f += m["f"]; all_p += m["p"]; all_r += m["r"]
            per_type[t] = {"faithfulness": f, "context_precision": p,
                           "context_recall": r, "n": len(m["f"])}
            print(f"{t}: faithfulness={f:.3f} precision={p:.3f} recall={r:.3f}", flush=True)
    summary = {
        "n": len(rows),
        "sampling": sampling_desc,
        "types_covered": sorted(covered),
        "overall": {
            "faithfulness": sum(all_f) / len(all_f),
            "context_precision": sum(all_p) / len(all_p),
            "context_recall": sum(all_r) / len(all_r),
        },
        "per_type": per_type,
    }
    print(f"整体: faithfulness={summary['overall']['faithfulness']:.3f} "
          f"precision={summary['overall']['context_precision']:.3f} "
          f"recall={summary['overall']['context_recall']:.3f}", flush=True)
    paths = artifacts.save_run("qa_ragas", rows, summary, out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="QA 语料 RAGAS 评估（逐题落盘）")
    p.add_argument("n", nargs="?", type=int, default=300, help="跑前 N 题")
    p.add_argument("--per-type", type=int, default=None,
                   help="每类取 N 题（分层抽样，优于取前缀）")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR, help="落盘目录")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    main(_args.n, per_type=_args.per_type, out_dir=_args.out)
