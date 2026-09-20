from rag.embeddings import Embedder


def test_embedder_shapes():
    emb = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", dim=384)
    vecs = emb.embed(["hello world", "goodbye world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
