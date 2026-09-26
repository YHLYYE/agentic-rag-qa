"""共享 LLM 包装（OpenAI 兼容协议 = DeepSeek）。

原先 app.py / run_qa_eval.py / run_topk_sweep.py / run_topk_sweep_fast.py / run_graph.py
各写了一份几乎相同的包装类，这里合并为一处，保证 temperature、strip、模型选择行为一致。
"""
from __future__ import annotations

import os
import time

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 30.0     # 秒：一次网络抖动不该让整个请求挂死
DEFAULT_MAX_RETRIES = 2    # 首次 + 2 次重试


class LLMCallError(RuntimeError):
    """LLM 调用在重试耗尽后仍失败。"""


class DeepSeekLLM:
    """temperature=0（与全部评估脚本一致），输出统一 strip。

    model 解析顺序：构造参数 > 环境变量 DEEPSEEK_MODEL > deepseek-chat。
    """

    def __init__(self, client, model: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 degrade: bool = False):
        self.client = client
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # degrade=True：重试耗尽后返回空串而不是抛异常。
        # 用于「链路里的一环」——生成失败时，让下游的引用闸门/CRAG 把它判成「答不了」，
        # 而不是让整个请求 500。
        self.degrade = degrade

    def complete(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:      # 超时 / 429 / 5xx / 连接重置
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * 2 ** attempt)   # 指数退避：0.5s, 1s, 2s
        if self.degrade:
            return ""
        raise LLMCallError(
            f"LLM 调用失败（已重试 {self.max_retries} 次）：{type(last_exc).__name__}: {last_exc}"
        ) from last_exc


def build_llm(degrade: bool = False):
    """从 .env / 环境变量构造（需要 DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY）。

    `degrade=True` 用于**交互入口**：LLM 打不通时返回空答案，让下游的引用闸门
    把它判成「答不了」，而不是把整个请求打成 500。评估脚本保持 False（要能看见失败）。
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        # 关掉 SDK 自带重试：否则它和本类的重试**叠乘** ——
        # 实测 1s 超时 + 1 次重试会拖到 9.8s。重试策略只留一处。
        max_retries=0,
    )
    return DeepSeekLLM(client, degrade=degrade)
