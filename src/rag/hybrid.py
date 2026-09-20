from models import RetrievedChunk


def merge_and_rerank(result_lists: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    for lst in result_lists:
        for rc in lst:
            if rc.chunk.chunk_id not in by_id or rc.score > by_id[rc.chunk.chunk_id].score:
                by_id[rc.chunk.chunk_id] = rc
    merged = sorted(by_id.values(), key=lambda x: -x.score)
    return merged[:top_k]
