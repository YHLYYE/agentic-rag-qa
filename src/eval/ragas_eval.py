from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall


def run_ragas(questions: list[str], answers: list[str],
              contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS faithfulness/context metrics. Requires the `ragas` package
    (installed during the evaluation integration step, not yet)."""
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(ds, metrics=[faithfulness, context_precision, context_recall])
    return dict(result)
