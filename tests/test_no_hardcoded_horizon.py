"""Tests verifying horizon text is never hardcoded to '30 min' or '30 minutes'."""

from pathlib import Path
import re


def test_no_hardcoded_horizon_in_web_and_docs():
    root = Path(__file__).resolve().parent.parent

    target_files = [
        root / "src" / "web" / "index.html",
        root / "src" / "web" / "app.js",
        root / "src" / "web" / "style.css",
        root / "docs" / "user_guide.md",
    ]

    # Pattern for literal 30 min / 30 minutes
    pattern = re.compile(r"\b30\s*min(utes)?\b", re.IGNORECASE)

    for fpath in target_files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        matches = []
        for line_num, line in enumerate(text.splitlines(), start=1):
            if "illustrative" in line.lower() or "example" in line.lower():
                continue
            if pattern.search(line):
                matches.append((line_num, line.strip()))

        assert not matches, f"Hardcoded '30 min' found in {fpath}: {matches}"

