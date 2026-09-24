import os
import tempfile

from ingest.index import detect_doc_changes


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
