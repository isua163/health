"""Make the bundled analysis package importable for every test invocation.

Tests must pass when a single file is selected and must not depend on another test
module having modified ``sys.path`` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
