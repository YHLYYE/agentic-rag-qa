import pickle
from pathlib import Path
import faiss
import numpy as np

from rag.embeddings import Embedder


def build_index(chunks, embedder: Embedder, index_dir: str) -> None:
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = [c.text for c in chunks]
    vecs = np.array(embedder.embed(texts), dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
