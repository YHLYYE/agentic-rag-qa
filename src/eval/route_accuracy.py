"""路由准确率：LLM 意图分类 vs 三类题的真实标签。"""
import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

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


def main() -> None:
    with open("data/qa/eval_set.pkl", "rb") as f:
        eval_set = pickle.load(f)
    llm = _LLM()
    correct = 0
    confusion = {}
    for item in eval_set:
        out = route_node({"question": item["question"]}, llm=llm)
        predicted = out["route_decision"]["semantic_intent"]
        gold = item["type"]
        if predicted == gold:
            correct += 1
        else:
            confusion[(gold, predicted)] = confusion.get((gold, predicted), 0) + 1
    print(f"路由准确率: {correct}/{len(eval_set)} = {correct / len(eval_set):.3f}", flush=True)
    if confusion:
        print("混淆(真实→预测):", confusion, flush=True)


if __name__ == "__main__":
    main()
