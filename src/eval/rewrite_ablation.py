"""改写消融：多轮追问场景下，「改写」到底有没有用、错多少。

为什么做这个：改写是**无人监督的一环** —— 它错了整个检索就偏了，而链路察觉不到。
所以我先量化「它靠不靠得住」，再决定要不要写兜底逻辑（凭感觉写兜底可能写了死代码）。

三路对比（同一检索器、同一批题、同一套 HitRate@5）：
    A 原始完整问题   → **上界**（等价于"完美改写"，检验任务本身可解）
    B 指代型追问     → **不改写的下界**（检验"不改写会掉多少"）
    C 改写后的追问   → **实际方案**

另报**改写保真率**：LLM 判官判「改写后的问题与原始问题是否在问同一件事」。

用法：
    python -m eval.rewrite_ablation --n 20
"""
from __future__ import annotations

import argparse

from eval import artifacts, retrieval_metrics as rm, sampling

FU = ("给定一个问题和它的答案，生成一个**像真人多轮对话第二问**的追问："
      "用「它 / 这个 / 那位」等代词指代关键实体，**不要重复实体名字**。"
      "只输出追问本身，不要解释。\n\n"
      "问题：{q}\n答案：{a}\n\n追问：")

# 注意：不能拿「改写后」和「原始问题」比 —— 追问本来就不是原问题（原问题可能是 A/B 比较，
# 追问只问其中一个实体的某个属性）。正确的对照是「追问 + 对话历史」：
# 改写有没有把「它/这个」正确解析到历史里的实体。
JUDGE = ("下面是多轮对话历史、用户带指代的追问、以及系统的改写结果。"
         "请判断：改写结果是否**正确还原了追问想问的东西**"
         "（即把「它/这个/那位」正确解析成了历史里的实体，且没有歪曲意图）。"
         "只回答「是」或「否」。\n\n对话历史：{h}\n用户追问：{q}\n改写结果：{r}\n")


def _is_yes(text: str) -> bool:
    return (text or "").strip().lower().startswith(("是", "yes", "y", "true"))


def main(n: int = 20, per_type: int | None = None, topk: int = 5,
         out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    from graph.nodes import rewrite_node
    from llm import DeepSeekLLM

    load_dotenv()
    client = OpenAI(base_url=os.environ["DEEPSEEK_BASE_URL"],
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    llm = DeepSeekLLM(client)

    eval_set = rm.load_eval_set()
    picked = (sampling.stratified_sample(eval_set, per_type) if per_type
              else eval_set[:n])
    retrieve = rm.build_retriever("hybrid")
    print(f"改写消融：n={len(picked)}  检索器=hybrid  k={topk}", flush=True)

    def hit(question: str, ground_truth: str) -> bool:
        return any(rm.answer_in_text(ground_truth, x.chunk.text)
                   for x in retrieve(question, topk))

    rows = []
    for i, item in enumerate(picked, 1):
        q, gt = item["question"], item["ground_truth"]
        follow_up = llm.complete(FU.format(q=q, a=gt)).strip()
        out = rewrite_node({"question": follow_up,
                            "history": [{"question": q, "answer": gt}]}, llm=llm)
        rewritten = out.get("question", follow_up)
        faithful = _is_yes(llm.complete(JUDGE.format(
            h=f"用户：{q}\n助手：{gt}", q=follow_up, r=rewritten)))
        rows.append({
            "question": q, "follow_up": follow_up, "rewritten": rewritten,
            "rewrite_faithful": faithful,
            "hit_original": hit(q, gt),          # A 上界
            "hit_follow_up": hit(follow_up, gt),  # B 不改写
            "hit_rewritten": hit(rewritten, gt),  # C 实际方案
        })
        r = rows[-1]
        print(f"  [{i}/{len(picked)}] 保真={faithful}  A={r['hit_original']} "
              f"B={r['hit_follow_up']} C={r['hit_rewritten']}  {follow_up[:26]}", flush=True)

    total = len(rows)
    summary = {
        "n": total, "retriever": "hybrid", "topk": topk,
        "hit_original(prompt上界)": sum(r["hit_original"] for r in rows) / total,
        "hit_follow_up(不改写)": sum(r["hit_follow_up"] for r in rows) / total,
        "hit_rewritten(实际方案)": sum(r["hit_rewritten"] for r in rows) / total,
        "rewrite_faithful_rate": sum(r["rewrite_faithful"] for r in rows) / total,
    }
    print("\n=== 改写消融结果 ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<26} {v:.4f}")
    print(f"  => 改写相对不改写的增益: "
          f"{summary['hit_rewritten(实际方案)'] - summary['hit_follow_up(不改写)']:+.4f}"
          f"   与上界的差距: "
          f"{summary['hit_original(prompt上界)'] - summary['hit_rewritten(实际方案)']:+.4f}")
    paths = artifacts.save_run("rewrite_ablation", rows, summary, out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}")
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="多轮改写消融")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    main(n=a.n, per_type=a.per_type, topk=a.topk, out_dir=a.out)
