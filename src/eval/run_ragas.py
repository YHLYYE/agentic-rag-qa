"""Run RAGAS (faithfulness / context precision / recall) with DeepSeek as judge."""
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

from eval.ragas_patch import patch
patch()  # must run before importing ragas

from ragas import evaluate
from ragas.metrics.collections import faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset, load_dataset

from rag.embeddings import Embedder
from rag.dense import DenseRetriever
from rag.bm25 import BM25Retriever
from rag.hybrid import merge_and_rerank


def _generate(chat, question: str, context: str) -> str:
    prompt = f"根据以下上下文回答问题，尽量简洁。\n\n问题：{question}\n\n上下文：\n{context}"
    return chat.invoke(prompt).content


def main(index_dir: str = "data/index", n: int = 15) -> None:
    load_dotenv()
    chat = ChatOpenAI(
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model=os.environ["DEEPSEEK_MODEL"],
        temperature=0,
    )
    judge = LangchainLLMWrapper(chat)

    with open(Path(index_dir) / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = Embedder(model_name="BAAI/bge-m3", dim=1024)
    dense = DenseRetriever(chunks, embedder, str(Path(index_dir) / "faiss.index"))
    bm25 = BM25Retriever(chunks)

    ds = load_dataset("PatronusAI/financebench", split="train")
    questions = list(ds["question"][:n])
    answers = list(ds["answer"][:n])

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q, gt in zip(questions, answers):
        rc = merge_and_rerank([dense.retrieve(q, 5), bm25.retrieve(q, 5)], top_k=5)
        ctx = [x.chunk.text for x in rc]
        gen = _generate(chat, q, "\n".join(ctx))
        rows["question"].append(q)
        rows["answer"].append(gen)
        rows["contexts"].append(ctx)
        rows["ground_truth"].append(gt)

    eval_ds = Dataset.from_dict(rows)
    result = evaluate(eval_ds, metrics=[faithfulness, context_precision, context_recall], llm=judge)
    print(result)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
