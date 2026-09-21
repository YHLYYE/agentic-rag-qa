"""Self-implemented RAGAS metrics (faithfulness / context precision / recall).

Uses DeepSeek (OpenAI-compatible) as the judge LLM, avoiding the ragas package
and its version conflicts with langgraph/langchain-core.
"""
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank


def _llm() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )


def _ask(client: OpenAI, prompt: str) -> str:
    r = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()


def faithfulness(client, question: str, answer: str, context: str) -> float:
    """RAGAS 细粒度 faithfulness：拆句 → 逐句判断是否被上下文支持。"""
    # 第1步：把答案拆成一句句独立断言
    decomp = _ask(client, (
        "把以下答案拆成一句句独立的事实断言，每句一行，只输出断言本身，"
        "不要编号、不要解释。\n\n答案：\n" + answer
    ))
    claims = [c.strip() for c in decomp.split("\n")
              if c.strip() and not c.strip().startswith(("#", "-", "*", "1.", "2.", "3."))]
    if not claims:
        return 0.0
    # 第2步：逐句判断是否被上下文支持
    supported = 0
    for claim in claims:
        verdict = _ask(client, (
            "判断以下断言是否被上下文支持。只回答「是」或「否」。\n\n"
            f"断言：{claim}\n\n上下文：\n{context}"
        ))
        if verdict.strip().lower().startswith(("是", "yes", "y", "true", "支持")):
            supported += 1
    return supported / len(claims)


def context_precision(client, question: str, context: str) -> float:
    prompt = (
        "给定问题和检索到的上下文，判断上下文中有多少比例的内容对回答问题是有用的（精准度）。\n\n"
        f"问题：{question}\n\n上下文：\n{context}\n\n只输出一个 0~1 的小数。"
    )
    try:
        return float(_ask(client, prompt))
    except ValueError:
        return 0.0


def context_recall(client, question: str, context: str, ground_truth: str) -> float:
    """RAGAS 细粒度 context recall：拆标准答案成信息点 → 逐点判断能否从上下文找到依据。"""
    decomp = _ask(client, (
        "把以下标准答案拆成一句句独立的信息点，每句一行，只输出信息点本身，"
        "不要编号、不要解释。\n\n标准答案：\n" + ground_truth
    ))
    points = [p.strip() for p in decomp.split("\n")
              if p.strip() and not p.strip().startswith(("#", "-", "*", "1.", "2.", "3."))]
    if not points:
        return 0.0
    attributed = 0
    for p in points:
        verdict = _ask(client, (
            "以下信息点能否从上下文中找到依据？只回答「是」或「否」。\n\n"
            f"信息点：{p}\n\n上下文：\n{context}"
        ))
        if verdict.strip().lower().startswith(("是", "yes", "y", "true", "支持")):
            attributed += 1
    return attributed / len(points)


def main(index_dir: str = "data/index", n: int = 30) -> None:
    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)

    ds = load_dataset("PatronusAI/financebench", split="train")
    questions = list(ds["question"][:n])
    answers = list(ds["answer"][:n])

    client = _llm()
    f_scores, p_scores, r_scores = [], [], []
    for i, (q, gt) in enumerate(zip(questions, answers), 1):
        rc = merge_and_rerank([dense.retrieve(q, 5), bm25.retrieve(q, 5)], top_k=5)
        ctx = "\n".join(x.chunk.text for x in rc)
        gen_prompt = (
            f"根据以下上下文回答问题，尽量简洁。\n\n问题：{q}\n\n上下文：\n{ctx}"
        )
        ans = _ask(client, gen_prompt)
        f_scores.append(faithfulness(client, q, ans, ctx))
        p_scores.append(context_precision(client, q, ctx))
        r_scores.append(context_recall(client, q, ctx, gt))
        print(f"[{i}/{n}] f={f_scores[-1]:.2f} p={p_scores[-1]:.2f} r={r_scores[-1]:.2f}", flush=True)

    print("\n=== 自实现 RAGAS 指标 (n=%d) ===" % n, flush=True)
    print(f"faithfulness:      {sum(f_scores)/len(f_scores):.3f}", flush=True)
    print(f"context_precision: {sum(p_scores)/len(p_scores):.3f}", flush=True)
    print(f"context_recall:    {sum(r_scores)/len(r_scores):.3f}", flush=True)


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    index_dir = argv[0] if argv else "data/index"
    n = int(argv[1]) if len(argv) > 1 else 30
    main(index_dir, n)
