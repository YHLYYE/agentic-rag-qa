from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, dim: int, normalize: bool = True):
        self.model_name = model_name
        self.dim = dim
        self.normalize = normalize
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=self.normalize,
            show_progress_bar=True, batch_size=16
        )
        return [v.tolist() for v in vecs]
