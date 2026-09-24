"""用 httpx 直接下载 QA 数据集的 parquet 文件（绕开 huggingface_hub 的 HEAD 请求问题）。"""
import os
from pathlib import Path

import httpx

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE = "https://hf-mirror.com"


def _download(url: str, dest: Path) -> int:
    r = httpx.get(url, timeout=600, follow_redirects=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return len(r.content)


def _list_files(repo: str) -> list[str]:
    r = httpx.get(f"{BASE}/api/datasets/{repo}", timeout=60, follow_redirects=True)
    r.raise_for_status()
    return [f["rfilename"] for f in r.json()["siblings"]]


def _pick_validation(files: list[str], prefix: str) -> list[str]:
    """选 prefix 下 validation 开头的 parquet 文件（按大小排序，取前 N 个 shard）。"""
    cands = [f for f in files if f.startswith(prefix) and "validation" in f and f.endswith(".parquet")]
    return sorted(cands)


def main() -> None:
    out = Path("data/raw_qa")
    out.mkdir(parents=True, exist_ok=True)

    targets = [
        ("hotpotqa/hotpot_qa", "distractor/", "hotpotqa_distractor_val"),
        ("trivia_qa", "rc.web/", "trivia_rcweb_val"),
    ]
    for repo, prefix, local_base in targets:
        try:
            files = _list_files(repo)
            vals = _pick_validation(files, prefix)
            if not vals:
                # 可能前缀不对，列出所有 validation
                vals = [f for f in files if "validation" in f and f.endswith(".parquet") and prefix.split("/")[0] in f]
            print(f"{repo}: 找到 {len(vals)} 个 validation shard: {vals[:5]}")
            for i, fname in enumerate(vals[:2]):  # 最多下 2 个 shard
                url = f"{BASE}/datasets/{repo}/resolve/main/{fname}"
                dest = out / f"{local_base}_{i}.parquet"
                if dest.exists():
                    print(f"  跳过 {fname}（已存在）")
                    continue
                size = _download(url, dest)
                print(f"  下载 {fname} → {size//1024}KB")
        except Exception as e:
            print(f"{repo}: 失败 {type(e).__name__} {str(e)[:120]}")


if __name__ == "__main__":
    main()
