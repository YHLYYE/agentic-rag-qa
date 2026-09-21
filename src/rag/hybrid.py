from models import RetrievedChunk


def merge_and_rerank(result_lists: list[list[RetrievedChunk]], top_k: int,
                     k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: rank-based merge, robust to differing score scales."""
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for lst in result_lists:
        for rank, rc in enumerate(lst):
            cid = rc.chunk.chunk_id
            chunk_by_id.setdefault(cid, rc)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return [chunk_by_id[cid] for cid, _ in ranked[:top_k]]
