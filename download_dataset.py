"""
Download the Kaggle Fake and Real News dataset into ./dataset.

Usage
-----
    python download_dataset.py

Requires the optional `kagglehub` package (`pip install kagglehub`).
If automatic download is unavailable, follow the manual instructions printed
on failure or in README.md. This script never generates synthetic data.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
FILES = ("Fake.csv", "True.csv")
MANUAL_HELP = """
Manual download
---------------
1. Open https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click "Download" (about 116 MB).
3. Extract the archive and copy Fake.csv and True.csv into:
   {directory}
4. Re-run:  python train_model.py
"""


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = [name for name in FILES if (DATASET_DIR / name).exists()]
    if len(existing) == len(FILES):
        sizes = {name: (DATASET_DIR / name).stat().st_size for name in FILES}
        print("Dataset already present:")
        for name, size in sizes.items():
            print(f"  dataset/{name}  ({size / 1024 / 1024:.1f} MB)")
        return 0

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Install it with: pip install kagglehub")
        print(MANUAL_HELP.format(directory=DATASET_DIR))
        return 1

    print("Downloading clmentbisaillon/fake-and-real-news-dataset via kagglehub …")
    try:
        source = Path(kagglehub.dataset_download(
            "clmentbisaillon/fake-and-real-news-dataset"
        ))
    except Exception as exc:
        print(f"Download failed: {exc}")
        print(MANUAL_HELP.format(directory=DATASET_DIR))
        return 1

    for name in FILES:
        src = source / name
        if not src.exists():
            print(f"Expected {name} inside {source}, but it was not found.")
            print(MANUAL_HELP.format(directory=DATASET_DIR))
            return 1
        shutil.copy2(src, DATASET_DIR / name)
        print(f"  copied {name} -> dataset/{name} "
              f"({(DATASET_DIR / name).stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nDone. Next steps:")
    print("  python preprocess.py      # optional: build corpus cache + tokenizer")
    print("  python train_model.py     # train LSTM and BiLSTM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
