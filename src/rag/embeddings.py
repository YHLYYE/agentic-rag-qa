from sentence_transformers import SentenceTransformer


def clamp_seq_length(model, max_seq_length: int | None):
    """把模型的最大序列长度压到指定值（None 表示不改）。

    为什么需要：bge-m3 默认 max_seq_length=8192，配合较大 batch 时
    attention mask 会按 batch × 8192² 展开 —— 实测直接 CUDA OOM（要 16GB）。
    检索场景通常 512~1024 token 足够，压一下既省显存又快得多。
    """
    if max_seq_length:
        model.max_seq_length = min(getattr(model, "max_seq_length", max_seq_length),
                                   max_seq_length)
    return model


class Embedder:
    def __init__(self, model_name: str, dim: int, normalize: bool = True,
                 device: str | None = None, show_progress: bool = True,
                 max_seq_length: int | None = None):
        self.model_name = model_name
        self.dim = dim
        self.normalize = normalize
        # device=None 自动选 GPU/CPU；device="cpu" 强制 CPU（演示用，避免显存不足）
        # show_progress=False 关掉进度条：它写 stderr，CLI 演示时会污染输出
        self.show_progress = show_progress
        self._model = SentenceTransformer(model_name, device=device)
        clamp_seq_length(self._model, max_seq_length)

    def embed(self, texts: list[str], show_progress: bool | None = None) -> list[list[float]]:
        bar = self.show_progress if show_progress is None else show_progress
        vecs = self._model.encode(
            texts, normalize_embeddings=self.normalize,
            show_progress_bar=bar, batch_size=16
        )
        return [v.tolist() for v in vecs]
