"""Dataset acquisition for the Bank Account Fraud (BAF) suite.

Phase 1 uses only ``Base.csv``. The Kaggle client authenticates *at import time*,
so we load ``.env`` and validate credentials BEFORE importing it.
"""
from __future__ import annotations

import hashlib
import os
import zipfile

from dotenv import load_dotenv

from . import config

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"


def _require_credentials() -> None:
    """Load .env and fail loudly if Kaggle env vars are missing.

    The kaggle client only reads KAGGLE_USERNAME and KAGGLE_KEY; a misnamed
    variable (e.g. KAGGLE_API_TOKEN) is silently ignored, so we check explicitly.
    """
    load_dotenv(config.ROOT / ".env")
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing Kaggle credentials: {', '.join(missing)}. "
            f"Set them in {config.ROOT / '.env'} (exact names KAGGLE_USERNAME and KAGGLE_KEY)."
        )


def sha256(path) -> str:
    """SHA-256 of a file, streamed so large CSVs don't blow up memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_baf(variant: str = "Base", force: bool = False):
    """Download a single BAF variant CSV into data/raw/. Idempotent.

    Returns the path to the extracted CSV.
    """
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    target = config.DATA_RAW / f"{variant}.csv"
    if target.exists() and not force:
        print(f"[data] {target.name} already present — skipping download.")
        return target

    _require_credentials()
    # Import only after credentials are set: authenticate() runs on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[data] Downloading {variant}.csv from {KAGGLE_DATASET} ...")
    # The suite stores each variant as a top-level CSV; fetch just the one file.
    api.dataset_download_file(
        KAGGLE_DATASET, file_name=f"{variant}.csv", path=str(config.DATA_RAW)
    )

    # Kaggle may deliver a .zip when the file is large; unzip if so.
    zipped = config.DATA_RAW / f"{variant}.csv.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(config.DATA_RAW)
        zipped.unlink()

    if not target.exists():
        raise FileNotFoundError(f"Expected {target} after download but it is missing.")
    print(f"[data] Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target
