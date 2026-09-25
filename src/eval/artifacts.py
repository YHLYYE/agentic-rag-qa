"""评估产物落盘：逐题明细 + 汇总，让任何汇总数字都能被追问到原始记录。

为什么要有这个模块：这些指标都有 LLM 判官参与，重跑不会得到完全一样的数字，
所以「能重新跑一遍」不等于「有证据」。唯一可靠的做法是跑一次就把逐题明细存下来，
以后引用这份存档，而不是每次重新生成一个新数字。
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = "data/reports/runs"


def _git_rev() -> str:
    """记录代码版本，便于判断存档是哪份代码跑出来的。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_run(name: str, detail_rows: list[dict], summary: dict,
             out_dir: str = DEFAULT_OUT_DIR,
             now: datetime | None = None) -> dict:
    """落盘一次评估：<ts>_<name>.jsonl（逐题）+ <ts>_<name>.summary.json（汇总）。

    同一秒内重复运行会加 -2/-3 后缀，绝不覆盖已有存档。
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{ts}_{name}"
    n = 1
    while (out / f"{stem}.jsonl").exists() or (out / f"{stem}.summary.json").exists():
        n += 1
        stem = f"{ts}_{name}-{n}"

    detail_path = out / f"{stem}.jsonl"
    summary_path = out / f"{stem}.summary.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    payload = {
        "name": name,
        "timestamp_utc": ts,
        "git_rev": _git_rev(),
        "n": len(detail_rows),
        "summary": summary,
        "detail_file": detail_path.name,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {"detail": str(detail_path), "summary": str(summary_path), "meta": payload}


def load_detail(path: str) -> list[dict]:
    """读回逐题明细，便于从原始记录重算汇总（而不是相信手抄的数字）。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
