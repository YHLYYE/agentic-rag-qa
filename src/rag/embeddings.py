from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, dim: int, normalize: bool = True,
                 device: str | None = None, show_progress: bool = True):
        self.model_name = model_name
        self.dim = dim
        self.normalize = normalize
        # device=None 自动选 GPU/CPU；device="cpu" 强制 CPU（演示用，避免显存不足）
        # show_progress=False 关掉进度条：它写 stderr，CLI 演示时会污染输出
        self.show_progress = show_progress
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str], show_progress: bool | None = None) -> list[list[float]]:
        bar = self.show_progress if show_progress is None else show_progress
        vecs = self._model.encode(
            texts, normalize_embeddings=self.normalize,
            show_progress_bar=bar, batch_size=16
        )
        return [v.tolist() for v in vecs]
