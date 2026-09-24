import hashlib
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


def detect_doc_changes(docs: dict[str, str], hash_file: str) -> dict:
    """增量索引的核心：用文档内容 hash 检测变化。

    docs: {doc_id: doc_text}
    返回 {doc_id: 'new' | 'changed' | 'unchanged'}，并保存新 hash。
    只有 new/changed 的文档需要重新切块+向量化。
    """
    hash_path = Path(hash_file)
    old_hashes = {}
    if hash_path.exists():
        with open(hash_path, "rb") as f:
            old_hashes = pickle.load(f)

    new_hashes = {}
    result = {}
    for doc_id, doc_text in docs.items():
        h = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
        new_hashes[doc_id] = h
        if doc_id not in old_hashes:
            result[doc_id] = "new"
        elif old_hashes[doc_id] != h:
            result[doc_id] = "changed"
        else:
            result[doc_id] = "unchanged"

    with open(hash_path, "wb") as f:
        pickle.dump(new_hashes, f)
    return result
