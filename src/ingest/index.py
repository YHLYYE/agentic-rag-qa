import hashlib
import pickle
from pathlib import Path
import faiss
import numpy as np

from rag.embeddings import Embedder


def build_index(chunks, embedder: Embedder, index_dir: str,
                incremental: bool = True) -> dict:
    """建 Faiss 索引。

    incremental=True 时复用**未变 chunk 的向量**：`chunk_id = sha1(来源|页|节|序号)`，
    所以内容没变的 chunk 会保持同一个 id，直接复用上次算好的向量，**只对新 chunk 调模型**。
    这就是「文档变更时只重新处理变化的部分」的落地（比按整份文档重算更细）。
    """
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache_path = out / "emb_cache.pkl"
    cached: dict[str, list[float]] = {}
    if incremental and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

    reused = [c for c in chunks if c.chunk_id in cached]
    todo = [c for c in chunks if c.chunk_id not in cached]
    fresh = embedder.embed([c.text for c in todo]) if todo else []
    vec_by_id = {c.chunk_id: cached[c.chunk_id] for c in reused}
    vec_by_id.update({c.chunk_id: v for c, v in zip(todo, fresh)})

    vecs = np.array([vec_by_id[c.chunk_id] for c in chunks], dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    if incremental:
        with open(cache_path, "wb") as f:
            pickle.dump(vec_by_id, f)

    stats = {"total": len(chunks), "reused": len(reused), "embedded": len(todo)}
    print(f"索引完成: 复用 {stats['reused']} / 新算 {stats['embedded']}（共 {stats['total']}）")
    return stats


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
