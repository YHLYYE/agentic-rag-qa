"""端到端跑一次完整的 LangGraph 五节点链路（唯一的人可运行入口）。

链路：route(LLM 意图路由) -> retrieve(多路检索) -> critique(检索批判)
      -> generate(生成候选答案+引用) -> verify(引用硬闸门) -> END / 回退重查

用法（在仓库根目录）：

    # 有 DeepSeek key（.env 或环境变量）-> 走完整的 LLM 意图路由
    $env:PYTHONPATH="src"; $env:HF_ENDPOINT="https://hf-mirror.com"
    python -m run_graph "谁发明了电话？"

    # 离线跑完整链路：不调任何模型接口，用抽取式占位模型代替生成
    python -m run_graph "谁发明了电话？" --no-llm

    # 排障：打印完整 state
    python -m run_graph "谁发明了电话？" --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# 让 `python src/run_graph.py` 与 `python -m run_graph` 都能找到 src 下的模块
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from graph.build import build_graph

DEFAULT_INDEX_DIR = "data/qa/index"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_EMBED_DIM = 1024

# 语料档案：把「索引目录 + 分词器 + 是否有标准答案」收成一处可选项。
# 关键点：**分词器必须随语料走** —— 英文分词器在中文上几乎不产生 token（实测 nDCG@10 0.0498 vs 0.4749）。
CORPORA = {
    "en_qa": {
        "index_dir": "data/qa/index",
        "tokenizer": "en",
        "label": "英文 QA benchmark（HotpotQA / TriviaQA）",
        "has_answers": True,
    },
    "zh_t2": {
        "index_dir": "data/zh/T2Ranking/index",
        "tokenizer": "zh",
        "label": "T2Ranking 检索子集（真实中文查询；**无标准答案**，只能看检索与忠实度）",
        "has_answers": False,
    },
}


def resolve_tokenizer(kind: str):
    from rag.bm25 import tokenize_no_stopwords, zh_tokenize
    if kind == "zh":
        return zh_tokenize
    if kind == "en":
        return tokenize_no_stopwords
    raise ValueError(f"未知分词器: {kind}")

# 生成 prompt 里每个上下文片段形如 "[chunk_id] 正文"
_CONTEXT_LINE = re.compile(r"^\[([a-f0-9]+)\]\s*(.+)$", re.MULTILINE)


class ExtractiveLLM:
    """离线占位模型：不调任何接口，把上下文第一段当答案并附上真实引用。

    用途：没有 API key 时也能把五节点链路（含引用硬闸门）完整跑一遍。
    它不是生成质量方案——`--no-llm` 演示出来的答案就是原文片段。
    """

    def complete(self, prompt: str) -> str:
        if "只输出类别名" in prompt:      # 路由 prompt：交回规则兜底
            return "factoid"
        m = _CONTEXT_LINE.search(prompt)
        if not m:
            return "查不到"
        chunk_id, text = m.group(1), m.group(2).strip()
        return f"{text[:300]} {{{{{chunk_id}}}}}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_graph",
        description="跑一次完整的 Agentic RAG 图链路并打印每个节点的状态",
    )
    p.add_argument("question", help="要问的问题")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                   help=f"索引目录（默认 {DEFAULT_INDEX_DIR}）")
    p.add_argument("--corpus", default="en_qa", choices=sorted(CORPORA),
                   help="语料档案：决定索引目录与分词器（zh_t2 为中文检索语料）")
    p.add_argument("--max-retry", type=int, default=2,
                   help="回退重查上限（默认 2）")
    p.add_argument("--no-llm", action="store_true",
                   help="离线模式：不调任何模型接口，用抽取式占位模型跑完整条图")
    p.add_argument("--rerank", action="store_true",
                   help="启用统一重排（粗排 top-20 → CrossEncoder 精排 → top-8）；"
                        "加载失败会自动降级为不重排")
    p.add_argument("--clarify", default=None,
                   help="问题触发澄清时，用这段补充信息续跑（演示 interrupt → resume 闭环）")
    p.add_argument("--thread", default="cli",
                   help="checkpoint 的会话标识（同一次多轮对话要一致）")
    p.add_argument("--json", action="store_true", help="打印完整 state（JSON）")
    return p


def initial_state(question: str, history: list | None = None) -> dict:
    """图的初始 state：字段与 AgenticRAGState 一一对应，缺一个节点就会读不到。

    `history` 是多轮对话历史（[{"question","answer"}]）—— 它只进生成 prompt，
    检索用的是 `rewrite_node` 改写出的独立问题。
    """
    return {
        "question": question,
        "history": list(history or []),
        "rewritten": False,
        "intent": "",
        "route_decision": {},
        "retrieved_chunks": [],
        "retrieval_verdict": "",
        "candidate_answer": "",
        "citations": [],
        "grounding_verdict": "",
        "final_answer": "",
        "retry_count": 0,
        "needs_clarify": False,
        "clarification": "",
    }


def run(question: str, retrievers: dict | None = None, llm=None, max_retry: int = 2,
        reranker=None, graph=None, thread_id: str | None = None,
        history: list | None = None) -> dict:
    """跑一次完整链路，返回最终 state。

    传入 `graph` + `thread_id` 时启用 checkpoint：问题触发 `interrupt()` 后会**真的暂停并留存**，
    可以用 `resume()` 带补充信息续跑（此前 checkpointer=None，只能暂停不能恢复）。
    """
    g = graph if graph is not None else build_graph(
        retrievers, llm, max_retry=max_retry, reranker=reranker)
    if thread_id:
        return g.invoke(initial_state(question, history),
                        {"configurable": {"thread_id": thread_id}})
    return g.invoke(initial_state(question, history))


def resume(clarification: str, graph, thread_id: str | None) -> dict:
    """从 `interrupt()` 处续跑。

    必须与首次调用**同一个 graph 实例 + 同一个 thread_id** —— 恢复依赖 checkpointer 里的存档，
    没有存档就无从恢复（所以这里显式报错，而不是静默返回一个看似正常的空状态）。
    """
    if not thread_id:
        raise ValueError("resume 需要 thread_id：interrupt 的恢复依赖 checkpointer 的存档")
    from langgraph.types import Command
    return graph.invoke(Command(resume=clarification),
                        {"configurable": {"thread_id": thread_id}})


def build_reranker(factory=None):
    """尝试加载精排模型；**失败时降级为 None（不精排）而不是抛异常**。

    为什么必须降级：本机 15.2GB 内存常驻应用占满，只剩 ~2GB 可用。bge-m3 与
    bge-reranker 同进程加载约需 4GB，靠 32GB 页面文件勉强撑住 —— 所以是**时好时坏**，
    实测遇到过 `OSError 1455 页面文件太小`。加载失败就打崩整个查询是不可接受的，
    重排只是锦上添花，不能成为单点故障。
    """
    try:
        if factory is None:
            from rag.reranker import Reranker
            factory = Reranker
        return factory()
    except Exception as e:  # 内存不足、模型缺失、依赖损坏都走这里
        print(f"[警告] 精排模型加载失败（{type(e).__name__}: {str(e)[:80]}），"
              f"已降级为不精排继续运行。", flush=True)
        return None


def format_trace(state: dict) -> list[str]:
    """把最终 state 渲染成「每个节点发生了什么」的可读追踪。"""
    route = state.get("route_decision") or {}
    chunks = state.get("retrieved_chunks") or []
    citations = state.get("citations") or []
    answer = state.get("candidate_answer") or ""

    lines = [
        f"[1] 路由       intent={state.get('intent')!r} "
        f"semantic_intent={route.get('semantic_intent')!r} "
        f"source={route.get('source')!r} "
        f"needs_clarify={state.get('needs_clarify')}",
        f"[2] 检索       命中 {len(chunks)} 个 chunk  "
        f"retry_count={state.get('retry_count')}",
    ]
    for i, c in enumerate(chunks[:5], 1):
        score = c.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        snippet = (c.get("text") or "")[:60].replace("\n", " ")
        lines.append(f"      #{i} {c.get('chunk_id')} score={score_text} {snippet}...")
    if len(chunks) > 5:
        lines.append(f"      ...（共 {len(chunks)} 个）")

    # 拒答路径：检索判定不可用 → 图直接跳过了生成节点，别假装生成过
    skipped_generate = not answer and not citations
    if skipped_generate:
        gen_line = "[4] 生成       未调用（检索判定不可用 → 直接拒答，不给幻觉留机会）"
    else:
        gen_line = (f"[4] 生成       候选答案 {len(answer)} 字，"
                    f"引用 {len(citations)} 个 chunk_id: {citations}")

    lines += [
        f"[3] 检索批判   retrieval_verdict={state.get('retrieval_verdict')!r}",
        gen_line,
        f"[5] 引用校验   grounding_verdict={state.get('grounding_verdict')!r}",
        "",
        f"最终答案：{state.get('final_answer')}",
    ]
    return lines


def load_retrievers(index_dir: str = DEFAULT_INDEX_DIR,
                    embed_model: str = DEFAULT_EMBED_MODEL,
                    embed_dim: int = DEFAULT_EMBED_DIM,
                    device: str | None = "cpu",
                    tokenizer: str = "en") -> dict:
    """加载离线索引 + 两个检索器，返回 {name: retrieve_fn} 供图使用。

    `tokenizer` 必须与语料匹配（"en" / "zh"）—— 用错会让 BM25 那一路基本失效。
    """
    import pickle

    from rag.bm25 import BM25Retriever
    from rag.dense import DenseRetriever
    from rag.embeddings import Embedder

    index_path = Path(index_dir)
    with open(index_path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    # 关掉进度条：它写 stderr，会让 PowerShell 把演示输出标成 NativeCommandError
    embedder = Embedder(model_name=embed_model, dim=embed_dim, device=device,
                        show_progress=False)
    dense = DenseRetriever(chunks, embedder, str(index_path / "faiss.index"))
    bm25 = BM25Retriever(chunks, tokenize=resolve_tokenizer(tokenizer))
    return {"dense": dense.retrieve, "bm25": bm25.retrieve}


# LLM 包装统一在 src/llm.py（原先 5 处各写了一份）；这里再导出一次，保持 `run_graph.DeepSeekLLM` 可用
from llm import DeepSeekLLM, build_llm  # noqa: E402  (需在 sys.path 处理之后导入)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    args = build_parser().parse_args(argv)
    llm = ExtractiveLLM() if args.no_llm else build_llm()
    reranker = build_reranker() if args.rerank else None

    print(f"问题：{args.question}")
    corpus = CORPORA[args.corpus]
    index_dir = corpus["index_dir"]
    print(f"配置：语料={args.corpus}（{corpus['label']}）  max_retry={args.max_retry}  "
          f"模型={'离线抽取式占位（--no-llm）' if args.no_llm else 'DeepSeek API'}")
    print(f"      index_dir={index_dir}  分词器={corpus['tokenizer']}")
    if not corpus["has_answers"]:
        print("提示：该语料**没有标准答案**，只能演示检索与忠实度，不能算答案正确率。")
    if args.rerank:
        print(f"精排：{'已启用（粗排 top-20 → 精排 top-8）' if reranker else '启用失败，已降级为不精排'}")
    print("加载检索器 ...", flush=True)
    retrievers = load_retrievers(index_dir, tokenizer=corpus["tokenizer"])

    # 带上 checkpointer，interrupt() 才是"可恢复的暂停"而不是"只能重跑"
    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph(retrievers, llm, max_retry=args.max_retry,
                        reranker=reranker, checkpointer=MemorySaver())
    state = run(args.question, graph=graph, thread_id=args.thread)

    print("\n=== 链路追踪 ===")
    if "__interrupt__" in state:
        print("[!] 问题过于模糊，图已在 clarify 节点暂停（interrupt）。")
        print(f"    澄清请求：{state['__interrupt__']}")
        if not args.clarify:
            print('    加 --clarify "<补充信息>" 即可演示「暂停 → 续跑」闭环。')
            return 0
        print(f'    → 用补充信息续跑："{args.clarify}"', flush=True)
        state = resume(args.clarify, graph, args.thread)
        print("    [恢复后] 已从 interrupt 处继续执行 ↓\n")
    for line in format_trace(state):
        print(line)

    if args.json:
        print("\n=== 完整 state ===")
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
