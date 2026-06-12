"""Feature engineering.

Phase 1 deliberately uses the raw BAF columns plus the ``*_missing`` indicator flags
created in ``preprocessing.clean()`` — no derived features. The baselines (especially
LightGBM) already extract strong signal from the raw inputs, and keeping the feature set
minimal makes the cost-sensitive analysis (the Phase-1 focus) easy to interpret.

This module is the home for engineered features in later phases (e.g. velocity ratios,
interaction terms, aggregations). Kept as a documented placeholder so the import path is
stable for downstream code.
"""
from __future__ import annotations

# Intentionally empty for Phase 1. See preprocessing.clean() for the missing-flag features.
