from eval.metrics import mrr, hit_rate, citation_hit_rate


def run_comparison(eval_set: list, graphs: dict) -> dict:
    """Run the three-group comparison (retrieval metrics).

    eval_set: list of dicts with keys 'gt_ids' (set of relevant doc ids).
    graphs: dict of group_name -> {'invoke': fn(question) -> dict with keys
        'ranked_ids', 'citations', 'retrieved_ids'}.
    Returns a dict group_name -> {mrr, hit_rate@5, citation_hit_rate}.
    """
    report = {}
    for group, cfg in graphs.items():
        rankings, cits, retrieved = [], [], []
        for q in eval_set:
            r = cfg["invoke"](q)
            rankings.append(r["ranked_ids"])
            cits.append(r["citations"])
            retrieved.append(set(r["retrieved_ids"]))
        report[group] = {
            "mrr": mrr(rankings, [q["gt_ids"] for q in eval_set]),
            "hit_rate@5": hit_rate(rankings, [q["gt_ids"] for q in eval_set], k=5),
            "citation_hit_rate": citation_hit_rate(cits, retrieved),
        }
    return report
