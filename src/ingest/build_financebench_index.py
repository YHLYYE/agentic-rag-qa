"""Build the FinanceBench search index from data/raw/*.pdf."""
from pathlib import Path

from ingest.parse import extract_pages
from ingest.chunk import chunk_text, detect_heading
from ingest.index import build_index
from rag.embeddings import Embedder


def main(raw_dir: str = "data/raw", index_dir: str = "data/index",
         model_name: str = "BAAI/bge-m3", dim: int = 1024) -> None:
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    print(f"PDFs found: {len(pdfs)}", flush=True)
    chunks = []
    for i, pdf in enumerate(pdfs, 1):
        pages = extract_pages(str(pdf))
        current_section = "GENERAL"
        for page_no, text in enumerate(pages, 1):
            heading = detect_heading(text)
            if heading:
                current_section = heading
            chunks.extend(chunk_text(text, source_doc=pdf.stem,
                                     section=current_section, page=page_no,
                                     chunk_size=256))
        print(f"[{i}/{len(pdfs)}] {pdf.stem}: {len(pages)} pages", flush=True)
    print(f"total chunks: {len(chunks)}", flush=True)

    print(f"loading embedding model {model_name} ...", flush=True)
    emb = Embedder(model_name=model_name, dim=dim)
    build_index(chunks, emb, index_dir)
    print(f"index written to {index_dir}", flush=True)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
