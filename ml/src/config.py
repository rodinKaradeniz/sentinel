"""Central paths and constants for Sentinel Phase 1.

Importable by scripts and (later) by serving/training code in other phases.
"""
from pathlib import Path

# Project root = two levels up from this file (ml/src/config.py -> repo root).
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# Global seed for reproducibility. Set everywhere randomness is used.
SEED = 42

# The BAF target column.
TARGET = "fraud_bool"
