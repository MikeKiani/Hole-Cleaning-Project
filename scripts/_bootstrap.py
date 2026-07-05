"""Add ``src/`` to ``sys.path`` so the numbered scripts run whether or not the
package has been pip-installed. Import this first in every script."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
