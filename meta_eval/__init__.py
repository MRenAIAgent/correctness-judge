# meta_eval - Meta-evaluation framework for correctness-judge

# Ensure correctness_judge is importable even when not pip-installed,
# by adding the src/ directory to sys.path if needed.
import sys
from pathlib import Path

try:
    import correctness_judge  # noqa: F401
except ImportError:
    _src = str(Path(__file__).resolve().parent.parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
