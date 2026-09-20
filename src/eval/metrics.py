def mrr(rankings: list[list[str]], ground_truth: list[set[str]]) -> float:
    if not rankings:
        return 0.0
    total = 0.0
    for ranked, gt in zip(rankings, ground_truth):
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in gt:
                total += 1.0 / i
                break
    return total / len(rankings)


def hit_rate(rankings: list[list[str]], ground_truth: list[set[str]], k: int = 5) -> float:
    if not rankings:
        return 0.0
    hits = sum(1 for ranked, gt in zip(rankings, ground_truth) if any(d in gt for d in ranked[:k]))
    return hits / len(rankings)


def citation_hit_rate(citations: list[list[str]], retrieved: list[set[str]]) -> float:
    if not citations:
        return 0.0
    ok = sum(1 for cits, ret in zip(citations, retrieved)
             if cits and all(c in ret for c in cits))
    return ok / len(citations)
