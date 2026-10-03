"""Pytest bootstrap: expose the project root on sys.path so tests can import
the ``src`` package regardless of the invocation directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))