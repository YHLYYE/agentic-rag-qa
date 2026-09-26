"""共享 LLM 包装：原先 5 个文件各写了一份，合并成一处后行为必须一致。"""
from types import SimpleNamespace

import llm


def _fake_client(content: str = "ok"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return client, calls


def test_complete_uses_temperature_zero_and_strips():
    client, calls = _fake_client("  answer  ")
    assert llm.DeepSeekLLM(client).complete("hi") == "answer"
    assert calls[0]["temperature"] == 0
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_model_defaults_to_deepseek_chat(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    client, calls = _fake_client()
    llm.DeepSeekLLM(client).complete("hi")
    assert calls[0]["model"] == "deepseek-chat"


def test_model_follows_env_var(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    client, calls = _fake_client()
    llm.DeepSeekLLM(client).complete("hi")
    assert calls[0]["model"] == "deepseek-reasoner"


def test_explicit_model_wins_over_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    client, calls = _fake_client()
    llm.DeepSeekLLM(client, model="custom").complete("hi")
    assert calls[0]["model"] == "custom"


# --- 生产可靠性：超时 / 重试 / 熔断降级（此前零超时零重试，一次网络抖动就 500）---

class _FlakyClient:
    """前 fail_times 次抛错，之后成功 —— 模拟网络抖动 / 429 限流。"""

    def __init__(self, fail_times=1, exc=None):
        self.fail_times = fail_times
        self.exc = exc or TimeoutError("read timeout")
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                if outer.calls <= outer.fail_times:
                    raise outer.exc
                from types import SimpleNamespace
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok"))])

        self.chat = SimpleNamespace(completions=_Completions())


def test_retries_transient_failures_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)   # 别真等
    client = _FlakyClient(fail_times=2)
    assert llm.DeepSeekLLM(client, max_retries=3).complete("hi") == "ok"
    assert client.calls == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = _FlakyClient(fail_times=99)
    import pytest
    with pytest.raises(llm.LLMCallError):
        llm.DeepSeekLLM(client, max_retries=2).complete("hi")
    assert client.calls == 3      # 首次 + 2 次重试


def test_degrade_mode_returns_empty_instead_of_crashing(monkeypatch):
    """降级模式：调用失败返回空串，由上层（CRAG/引用闸门）把它判成「答不了」。"""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = _FlakyClient(fail_times=99)
    assert llm.DeepSeekLLM(client, max_retries=1, degrade=True).complete("hi") == ""


def test_timeout_is_passed_to_the_api():
    client, calls = _fake_client()
    llm.DeepSeekLLM(client, timeout=7.5).complete("hi")
    assert calls[0]["timeout"] == 7.5
