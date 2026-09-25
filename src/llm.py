"""共享 LLM 包装（OpenAI 兼容协议 = DeepSeek）。

原先 app.py / run_qa_eval.py / run_topk_sweep.py / run_topk_sweep_fast.py / run_graph.py
各写了一份几乎相同的包装类，这里合并为一处，保证 temperature、strip、模型选择行为一致。
"""
from __future__ import annotations

import os

DEFAULT_MODEL = "deepseek-chat"


class DeepSeekLLM:
    """temperature=0（与全部评估脚本一致），输出统一 strip。

    model 解析顺序：构造参数 > 环境变量 DEEPSEEK_MODEL > deepseek-chat。
    """

    def __init__(self, client, model: str | None = None):
        self.client = client
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()


def build_llm():
    """从 .env / 环境变量构造（需要 DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY）。"""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    return DeepSeekLLM(client)
