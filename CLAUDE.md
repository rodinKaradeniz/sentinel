# CLAUDE.md — Sentinel

Guidance for Claude Code when working in this repo.

## What this is
Sentinel is a multi-phase ML platform for fraud / risk scoring, built as a learning +
portfolio project. **Phase 1 (ML baseline + cost-sensitive evaluation) is complete.**
Later phases add MLflow, FastAPI, monitoring, a Go control plane, and Kubernetes — so keep
real logic in importable `ml/src/` modules and keep scripts thin.

## How the user wants you to work
- **Step by step.** Do ONE logical step, show code/output, explain WHAT and WHY (especially
  ML judgment calls), then STOP and wait for explicit "go" before the next step.
- Name alternatives when there's a real choice (a metric, a split, an imbalance strategy)
  and say why you picked one.
- Keep explanations tight; the user has ML basics and is re-learning concepts.
- Be honest: surface mistakes and correct your own earlier claims when the data says so
  (we did exactly this with the cost-vs-F1 gap in Step 8/9).
- Ask rather than assume when ambiguous.

## Conventions
- Python 3.13 in `.venv`; dependencies pinned in `requirements.txt`. Some pins were bumped
  for 3.13 wheel availability (pyarrow 19.0.1, not 17.x).
- `ml/src/` = importable library (`config`, `data`, `preprocessing`, `features`, `models`,
  `metrics`). `ml/scripts/` = thin CLIs. `ml/configs/` = run config (YAML).
- Run scripts as modules from the repo root: `python -m ml.scripts.train`.
- Global `seed=42` and cost matrix live in `ml/src/config.py`.
- Reproducibility matters: seeds set, versions pinned, dataset SHA-256 recorded.
- Reports live in `reports/`; the EDA notebook is `notebooks/eda.ipynb`.
- LightGBM/RandomForest with `n_jobs=-1` emit harmless `ResourceTracker` cleanup noise on
  macOS/3.13 at process exit — filter it in shell output, don't "fix" it in code.

## Data
- Bank Account Fraud (BAF) `Base.csv` from Kaggle
  (`sgpjesus/bank-account-fraud-dataset-neurips-2022`), 1,000,000 rows, ~1.1% fraud.
- Acquire via `ml/src/data.py`; needs `KAGGLE_USERNAME` + `KAGGLE_KEY` in a `.env` at repo
  root (the kaggle client ignores any other variable name, e.g. `KAGGLE_API_TOKEN`).
- `.env`, `data/`, `models/`, and `personal_notes.txt` are gitignored — never commit them.
- Missing values are encoded as negatives; `preprocessing.clean()` handles them. The
  temporal split (train months 0–5, val 6, test 7) is the only valid split — never random.

## Phase 1 result (for context)
LightGBM won (test PR-AUC ≈ 0.21). The differentiator is the cost-sensitive threshold:
pick the cut that minimizes expected dollar cost (FN=$500, FP=$50), chosen on val and
evaluated on the sealed test month. See `reports/baseline_metrics.md`,
`reports/cost_sensitive_analysis.md`, and `reports/phase1_summary.md`.

## Don't (in Phase 1 scope)
- Don't add Docker / MLflow / API / infra yet — those are later phases.
- Don't over-engineer; clean and readable beats clever.
- Don't commit unless asked. The repo currently has no commits.
