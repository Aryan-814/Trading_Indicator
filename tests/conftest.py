"""Ensure src/ and project root are on sys.path for all tests."""
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
_src = _root / "src"
for p in [str(_root), str(_src)]:
    if p not in sys.path:
        sys.path.insert(0, p)

