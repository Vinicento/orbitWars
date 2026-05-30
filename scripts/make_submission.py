"""
Create Kaggle submission archive.

No argparse. Run from project root:
    python scripts/make_submission.py
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

OUT = Path("submissions/submission.tar.gz")
FILES = [
    Path("main.py"),
    Path("agents/__init__.py"),
    Path("agents/roi_expansion.py"),
    Path("agents/nearest_sniper.py"),
    Path("agents/passive.py"),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for file in FILES:
            if not file.exists():
                raise FileNotFoundError(file)
            tar.add(file, arcname=file.as_posix())

    print(f"Created: {OUT.resolve()}")
    print('Submit: kaggle competitions submit orbit-wars -f submissions/submission.tar.gz -m "roi expansion"')


if __name__ == "__main__":
    main()
