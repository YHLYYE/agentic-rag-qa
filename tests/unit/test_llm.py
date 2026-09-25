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
