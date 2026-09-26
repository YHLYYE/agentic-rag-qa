import os
import tempfile

import numpy as np

from ingest.index import build_index, detect_doc_changes
from models import Chunk


class _CountingEmbedder:
    """数一数到底嵌了多少条 —— 增量索引的价值就是「这个数变小」。"""

    def __init__(self):
        self.calls = 0
        self.embedded = 0

    def embed(self, texts):
        self.calls += 1
        self.embedded += len(texts)
        return [np.ones(4, dtype="float32") * (i + 1) for i in range(len(texts))]


def _chunks(texts):
    return [Chunk(f"id{i}", t, "doc", "s", 1) for i, t in enumerate(texts)]


def test_first_build_embeds_everything(tmp_path):
    emb = _CountingEmbedder()
    stats = build_index(_chunks(["a", "b", "c"]), emb, str(tmp_path / "idx"))
    assert stats == {"total": 3, "reused": 0, "embedded": 3}


def test_second_build_reuses_unchanged_chunks(tmp_path):
    idx = str(tmp_path / "idx")
    build_index(_chunks(["a", "b", "c"]), _CountingEmbedder(), idx)

    emb = _CountingEmbedder()
    stats = build_index(_chunks(["a", "b", "c"]), emb, idx)
    assert stats["reused"] == 3 and stats["embedded"] == 0
    assert emb.calls == 0                      # 一条都没重新嵌


def test_only_new_chunks_get_embedded(tmp_path):
    """文档变了 → 只有那几个新 chunk 会调模型，其余复用。"""
    idx = str(tmp_path / "idx")
    build_index(_chunks(["a", "b", "c"]), _CountingEmbedder(), idx)

    emb = _CountingEmbedder()
    stats = build_index(_chunks(["a", "b", "c", "d", "e"]), emb, idx)
    assert stats == {"total": 5, "reused": 3, "embedded": 2}


def test_detect_doc_changes_first_build():
    tmp = tempfile.mktemp(suffix=".pkl")
    docs = {"a": "hello world", "b": "foo bar", "c": "baz qux"}
    r = detect_doc_changes(docs, tmp)
    assert r == {"a": "new", "b": "new", "c": "new"}
    os.remove(tmp)


def test_detect_doc_changes_incremental():
    tmp = tempfile.mktemp(suffix=".pkl")
    docs1 = {"a": "hello world", "b": "foo bar", "c": "baz qux"}
    detect_doc_changes(docs1, tmp)

    # 第二次：a 改了、b 没变、c 删除、d 新增
    docs2 = {"a": "hello world CHANGED", "b": "foo bar", "d": "new doc"}
    r = detect_doc_changes(docs2, tmp)
    assert r["a"] == "changed"
    assert r["b"] == "unchanged"
    assert r["d"] == "new"
    assert "c" not in r  # 删除的文档不在结果里
    os.remove(tmp)
