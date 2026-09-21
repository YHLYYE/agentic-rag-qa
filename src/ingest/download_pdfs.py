"""Download FinanceBench PDFs (84 unique filings) to data/raw/ concurrently."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def _download_one(doc_name: str, link: str, out: Path) -> tuple[str, str, int]:
    dest = out / f"{doc_name}.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        return doc_name, "skip", 0
    try:
        r = requests.get(link, headers=HEADERS, timeout=120, allow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return doc_name, "ok", len(r.content)
    except Exception as e:
        return doc_name, f"fail:{type(e).__name__}", 0


def main(raw_dir: str = "data/raw", workers: int = 8) -> None:
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("PatronusAI/financebench", split="train")
    seen: dict[str, str] = {}
    for doc_name, link in zip(ds["doc_name"], ds["doc_link"]):
        seen.setdefault(doc_name, link)

    print(f"total unique docs: {len(seen)}, workers: {workers}", flush=True)
    ok = fail = skip = 0
    tasks = [(n, l, out) for n, l in seen.items()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, t[0], t[1], t[2]) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            name, status, _ = f.result()
            if status == "ok":
                ok += 1
                print(f"[{i}/{len(seen)}] OK   {name}", flush=True)
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[{i}/{len(seen)}] FAIL {name}: {status}", flush=True)
    print(f"\ndone: ok={ok} fail={fail} skip={skip}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
