from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, dim: int, normalize: bool = True, device: str | None = None):
        self.model_name = model_name
        self.dim = dim
        self.normalize = normalize
        # device=None 自动选 GPU/CPU；device="cpu" 强制 CPU（演示用，避免显存不足）
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=self.normalize,
            show_progress_bar=True, batch_size=16
        )
        return [v.tolist() for v in vecs]
