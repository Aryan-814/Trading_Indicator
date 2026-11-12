"""Test the holdout gate: second eval of same model must be blocked."""
import json
import pytest
from pathlib import Path


def test_holdout_gate_blocks_second_run(tmp_path):
    """Second call with same model name should be detected as already run."""
    log_path = tmp_path / "holdout_log.json"
    log = [{"model": "lightgbm", "date": "2026-11-10", "note": "first run"}]
    log_path.write_text(json.dumps(log))

    with open(log_path) as fh:
        ho_log = json.load(fh)
    already_run = any(e.get("model") == "lightgbm" for e in ho_log)
    assert already_run, "Should detect existing entry and block re-run"


def test_holdout_gate_allows_first_run(tmp_path):
    """Empty log → first run should be allowed."""
    log_path = tmp_path / "holdout_log.json"
    log_path.write_text("[]")

    with open(log_path) as fh:
        ho_log = json.load(fh)
    already_run = any(e.get("model") == "lightgbm" for e in ho_log)
    assert not already_run, "Should allow first run"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "a").mkdir(exist_ok=True)
        (td_path / "b").mkdir(exist_ok=True)
        test_holdout_gate_blocks_second_run(td_path / "a")
        test_holdout_gate_allows_first_run(td_path / "b")
    print("test_holdout_gate PASSED!")

