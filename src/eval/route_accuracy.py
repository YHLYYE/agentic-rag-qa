"""路由准确率：LLM 意图分类 vs 三类题的真实标签（逐题落盘）。"""
import argparse
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

from eval import artifacts, sampling
from graph.nodes import route_node


class _LLM:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            base_url=os.environ["DEEPSEEK_BASE_URL"],
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content.strip()


def main(limit: int | None = None, per_type: int | None = None,
         out_dir: str = artifacts.DEFAULT_OUT_DIR) -> dict:
    with open("data/qa/eval_set.pkl", "rb") as f:
        all_items = pickle.load(f)
    if per_type:
        eval_set = sampling.stratified_sample(all_items, per_type)
        sampling_desc = {"mode": "stratified", "per_type": per_type}
    elif limit:
        eval_set = all_items[:limit]
        sampling_desc = {"mode": "prefix", "limit": limit}
    else:
        eval_set = all_items
        sampling_desc = {"mode": "all"}

    covered = {i["type"] for i in eval_set}
    if len(covered) == 1 and sampling_desc["mode"] != "all":
        print(f"[警告] 抽样只覆盖单一题型 {covered}，结论不可外推；建议改用 --per-type",
              flush=True)

    llm = _LLM()
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    per_class: dict[str, list[bool]] = {}
    rows = []
    for item in eval_set:
        out = route_node({"question": item["question"]}, llm=llm)
        predicted = out["route_decision"]["semantic_intent"]
        gold = item["type"]
        hit = predicted == gold
        correct += hit
        per_class.setdefault(gold, []).append(hit)
        if not hit:
            confusion[(gold, predicted)] = confusion.get((gold, predicted), 0) + 1
        rows.append({"question": item["question"], "gold": gold,
                     "predicted": predicted, "strategy": out["intent"],
                     "correct": hit})

    summary = {
        "accuracy": correct / len(eval_set),
        "correct": correct,
        "total": len(eval_set),
        "sampling": sampling_desc,
        "types_covered": sorted(covered),
        "per_class_accuracy": {k: sum(v) / len(v) for k, v in per_class.items()},
        "per_class_support": {k: len(v) for k, v in per_class.items()},
        # 元组 key 不能直接进 JSON，转成 "gold->predicted"
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
    }
    print(f"路由准确率: {correct}/{len(eval_set)} = {summary['accuracy']:.3f}", flush=True)
    print("分题型:", {k: round(v, 3) for k, v in summary["per_class_accuracy"].items()},
          flush=True)
    if confusion:
        print("混淆(真实→预测):", confusion, flush=True)
    paths = artifacts.save_run("route_accuracy", rows, summary, out_dir=out_dir)
    print(f"已落盘: {paths['detail']} | {paths['summary']}", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="路由准确率评估（逐题落盘）")
    p.add_argument("limit", nargs="?", type=int, default=None, help="只跑前 N 题")
    p.add_argument("--per-type", type=int, default=None,
                   help="每类取 N 题（分层抽样，优于取前缀）")
    p.add_argument("--out", default=artifacts.DEFAULT_OUT_DIR, help="落盘目录")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    main(limit=_args.limit, per_type=_args.per_type, out_dir=_args.out)
