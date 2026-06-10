# Dataset Version — Bank Account Fraud (BAF)

Recorded: 2026-06-10 (Phase 1, Step 2).

## Source
- Kaggle: `sgpjesus/bank-account-fraud-dataset-neurips-2022` (NeurIPS 2022, Feedzai, synthetic).
- Variant used in Phase 1: **Base.csv** (the realistic sample).

## File
- Path: `data/raw/Base.csv` (gitignored)
- Size: 213.4 MB
- Rows: 1,000,000
- Columns: 32
- SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`

## Target
- Column: `fraud_bool`
- Positives (fraud): 11,029
- Base fraud rate: 1.1029%

## Tooling note
- kaggle client 1.6.17 (pinned). Server reported 2.0.2 — the version warning is cosmetic;
  download/list endpoints work on 1.6.17. Pinned for reproducibility.
