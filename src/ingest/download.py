from pathlib import Path
from datasets import load_dataset


def download_financebench(raw_dir: str, split: str = "train") -> dict:
    """Download FinanceBench questions/answers + doc metadata.

    PDFs are fetched separately (SEC EDGAR / HF mirror). Returns the dataset
    rows so downstream can map doc_name -> pdf path."""
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("PatronusAI/financebench", split=split)
    return ds
