# Sentinel

An ML platform for fraud / risk scoring. Built in phases as a learning + portfolio project.

## Phase 1 — ML baseline + cost-sensitive evaluation (current)
Pure ML: establish baseline fraud-detection models on the Bank Account Fraud (BAF)
dataset and select decision thresholds by **minimizing expected dollar cost** rather
than maximizing F1. No API, Docker, or infra yet.

- Dataset: [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) — NeurIPS 2022, synthetic, tabular, highly imbalanced, with a temporal `month` field.
- Model: LightGBM (plus Logistic Regression and Random Forest baselines).
- Differentiator: cost-sensitive threshold analysis as fraud base rate shifts.

## Roadmap (later phases)
2. Experiment tracking (MLflow)
3. Serving API (FastAPI)
4. Monitoring
5. Go control plane
6. Kubernetes deployment

Code is structured so Phase 1 logic (`ml/src/`) can be imported and reused by later phases.

## Layout
```
data/raw/          # original BAF files (gitignored)
data/processed/    # train/val/test splits (gitignored)
ml/src/            # importable library code
ml/scripts/        # thin CLIs (train.py, evaluate.py)
ml/configs/        # run configs (model params, cost matrix)
notebooks/         # EDA only
reports/           # baseline_metrics.md
models/            # saved artifacts (gitignored)
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility
Global seed fixed in `ml/src/config.py`. Dataset version/hash recorded in `reports/`.
