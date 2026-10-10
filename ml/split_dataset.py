"""Dataset splitting entrypoint for ml/ pipeline (delegates to scripts.split_xbat)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_xbat import build_and_split_xbat, XBAT_CATEGORIES, HIGH_RISK_LITHIUM_CATEGORIES, XBAT_TO_FUTURAM_CATEGORY

__all__ = [
    "build_and_split_xbat",
    "XBAT_CATEGORIES",
    "HIGH_RISK_LITHIUM_CATEGORIES",
    "XBAT_TO_FUTURAM_CATEGORY",
]

if __name__ == "__main__":
    build_and_split_xbat()
