from rag.embeddings import Embedder, clamp_seq_length
import numpy as np


def test_embedder_shapes():
    emb = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", dim=384)
    vecs = emb.embed(["hello world", "goodbye world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def _embedder_with_fake_model(show_progress: bool = True):
    """不加载真实模型（__new__ 跳过 __init__），只测参数透传。"""
    emb = Embedder.__new__(Embedder)
    emb.model_name = "fake"
    emb.dim = 2
    emb.normalize = True
    emb.show_progress = show_progress
    calls = {}

    class _FakeModel:
        def encode(self, texts, **kwargs):
            calls.update(kwargs)
            return np.array([[0.0, 1.0] for _ in texts], dtype="float32")

    emb._model = _FakeModel()
    return emb, calls


def test_embedder_uses_instance_progress_setting():
    """CLI 演示要关掉进度条：它写 stderr，在 PowerShell 里会被当成报错。"""
    emb, calls = _embedder_with_fake_model(show_progress=False)
    emb.embed(["a"])
    assert calls["show_progress_bar"] is False


def test_embedder_progress_can_be_overridden_per_call():
    emb, calls = _embedder_with_fake_model(show_progress=False)
    emb.embed(["a"], show_progress=True)
    assert calls["show_progress_bar"] is True


def test_embedder_progress_defaults_to_on():
    emb, calls = _embedder_with_fake_model()
    emb.embed(["a"])
    assert calls["show_progress_bar"] is True


# --- max_seq_length 截断：bge-m3 默认 8192 会让 attention mask 撑爆显存 ---

class _FakeST:
    def __init__(self, msl):
        self.max_seq_length = msl


def test_clamp_seq_length_lowers_the_cap():
    m = clamp_seq_length(_FakeST(8192), 512)
    assert m.max_seq_length == 512


def test_clamp_seq_length_never_raises_the_cap():
    """模型本来就更短时不该被「拉长」。"""
    m = clamp_seq_length(_FakeST(256), 512)
    assert m.max_seq_length == 256


def test_clamp_seq_length_none_keeps_model_default():
    m = clamp_seq_length(_FakeST(8192), None)
    assert m.max_seq_length == 8192
