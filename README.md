# Sentinel

An ML platform for fraud / risk scoring. Built in phases as a learning + portfolio project.

## Phase 1 — ML baseline + cost-sensitive evaluation (complete)
Pure ML: establish baseline fraud-detection models on the Bank Account Fraud (BAF)
dataset and select decision thresholds by **minimizing expected dollar cost** rather
than maximizing F1. No API, Docker, or infra yet.

- Dataset: [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) — NeurIPS 2022, synthetic, tabular, highly imbalanced (~1.1% fraud), with a temporal `month` field.
- Models: Logistic Regression and Random Forest baselines; **LightGBM** the winner (test PR-AUC 0.21, ~14× over random).
- Differentiator: cost-sensitive threshold analysis — the cost-optimal cut beats the default 0.5 (which is *worse than doing nothing* here) and shifts as the fraud base rate drifts.

### Results
See [reports/baseline_metrics.md](reports/baseline_metrics.md) (final metrics + cost outcome),
[reports/model_comparison.md](reports/model_comparison.md), and
[reports/cost_sensitive_analysis.md](reports/cost_sensitive_analysis.md).

### Pipeline
```bash
python -m ml.scripts.train      # train on months 0-5, pick cost-optimal threshold on month 6, save artifact
python -m ml.scripts.evaluate   # evaluate the saved artifact on the sealed test month (7)
```
The data is acquired via the Kaggle API (`ml/src/data.py`); set `KAGGLE_USERNAME` and
`KAGGLE_KEY` in a `.env` file at the repo root first.

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
