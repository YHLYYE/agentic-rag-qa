from ingest.chunk import chunk_text, make_chunk_id


def test_chunk_text_splits_by_size():
    text = "word " * 200  # 200 words
    chunks = chunk_text(text, source_doc="doc.pdf", section="S1", page=3,
                        chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(c.source_doc == "doc.pdf" for c in chunks)
    assert all(c.section == "S1" for c in chunks)
    assert all(c.page == 3 for c in chunks)


def test_chunk_ids_are_unique_and_stable():
    text = "hello world " * 50
    a = chunk_text(text, source_doc="d", section="s", page=1)
    b = chunk_text(text, source_doc="d", section="s", page=1)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids))
