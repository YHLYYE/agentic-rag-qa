"""评估产物落盘：任何汇总数字都必须能追问到逐题原始记录。"""
import json
from pathlib import Path

from eval import artifacts


def _read_jsonl(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.strip().splitlines()]


def test_save_run_writes_one_json_object_per_line(tmp_path):
    rows = [{"i": 1, "q": "a"}, {"i": 2, "q": "b"}]
    res = artifacts.save_run("demo", rows, {"accuracy": 0.5}, out_dir=str(tmp_path))
    parsed = _read_jsonl(res["detail"])
    assert len(parsed) == 2
    assert parsed[0]["q"] == "a"


def test_save_run_summary_carries_metadata_for_reproducibility(tmp_path):
    res = artifacts.save_run("demo", [{"i": 1}], {"accuracy": 1.0},
                             out_dir=str(tmp_path))
    meta = json.loads(Path(res["summary"]).read_text(encoding="utf-8"))
    assert meta["name"] == "demo"
    assert meta["n"] == 1
    assert meta["summary"]["accuracy"] == 1.0
    assert meta["git_rev"]                    # 能对上代码版本
    assert meta["detail_file"].endswith(".jsonl")


def test_save_run_preserves_non_ascii(tmp_path):
    res = artifacts.save_run("demo", [{"q": "谁发明了电话？"}], {}, out_dir=str(tmp_path))
    assert "谁发明了电话" in Path(res["detail"]).read_text(encoding="utf-8")


def test_save_run_creates_missing_directory(tmp_path):
    target = tmp_path / "nested" / "runs"
    res = artifacts.save_run("demo", [], {}, out_dir=str(target))
    assert Path(res["detail"]).exists()
    assert Path(res["summary"]).exists()


def test_save_run_names_are_distinct_within_same_second(tmp_path):
    """同一秒跑两次不能互相覆盖（否则旧证据会被悄悄冲掉）。"""
    a = artifacts.save_run("x", [{"i": 1}], {}, out_dir=str(tmp_path))
    b = artifacts.save_run("x", [{"i": 2}], {}, out_dir=str(tmp_path))
    assert a["detail"] != b["detail"]
    assert Path(a["detail"]).exists() and Path(b["detail"]).exists()
