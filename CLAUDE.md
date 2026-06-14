# CLAUDE.md — Sentinel

Guidance for Claude Code when working in this repo.

## What this is
Sentinel is a multi-phase ML platform for fraud / risk scoring, built as a learning +
portfolio project. Keep real logic in importable `ml/src/` modules and keep scripts thin,
because later phases (MLflow, FastAPI, monitoring, a Go control plane, Kubernetes) reuse it.

---

## Current status  (volatile — overwrite as the project moves)
- **Phase 1 (ML baseline + cost-sensitive evaluation) is complete.** LightGBM won
  (test PR-AUC ≈ 0.21, ~14× the no-skill baseline of ~0.015). The differentiator is the
  cost-sensitive threshold: pick the cut that minimizes expected dollar cost, chosen on val
  and evaluated on the sealed test month. See `reports/phase1_summary.md`,
  `reports/baseline_metrics.md`, `reports/cost_sensitive_analysis.md`.
- **Cost matrix = a chosen modeling assumption, not ground truth:** FN=$500, FP=$50 (10:1).
  Defensible for account-opening fraud but not empirical — revisit / justify if challenged.
  Lives in `ml/src/config.py`.
- **Git:** no commits yet (the user commits only when they ask; offer, don't assume).
- **Next:** Phase 2 = experiment tracking (MLflow). Not started.

---

## How the user wants you to work  (durable)
- **Step by step.** Do ONE logical step, show code/output, explain WHAT and WHY (especially
  ML judgment calls), then STOP and wait for explicit "go" before the next step.
- Name alternatives when there's a real choice (a metric, a split, an imbalance strategy)
  and say why you picked one.
- Keep explanations tight; the user has ML basics and is re-learning concepts.
- Be honest: surface mistakes and correct earlier claims when the data says so. Example
  lesson worth keeping: we first overstated how much the cost-optimal threshold beats the
  F1-optimal one; the leakage-free numbers showed the gap is small at a 10:1 cost ratio and
  only widens as the cost asymmetry grows — so "cost-optimal wins" is about *principle and
  robustness across cost ratios*, not a fixed dollar saving.
- Ask rather than assume when ambiguous.

## Conventions  (durable)
- Python 3.13 in `.venv`; dependencies pinned in `requirements.txt`. Some pins were bumped
  for 3.13 wheel availability (pyarrow 19.0.1, not 17.x).
- `ml/src/` = importable library (`config`, `data`, `preprocessing`, `features`, `models`,
  `metrics`). `ml/scripts/` = thin CLIs. `ml/configs/` = run config (YAML).
- Run scripts as modules from the repo root: `python -m ml.scripts.train`.
- Global `seed=42` and the cost matrix live in `ml/src/config.py`.
- Reproducibility matters: seeds set, versions pinned, dataset SHA-256 recorded
  (`models/lightgbm_baseline.meta.json`).
- Reports live in `reports/`; the EDA notebook is `notebooks/eda.ipynb`.
- LightGBM/RandomForest with `n_jobs=-1` emit harmless `ResourceTracker` cleanup noise on
  macOS/3.13 at process exit — filter it in shell output, don't "fix" it in code.
- Don't commit unless asked. Don't add Docker / MLflow / API / infra ahead of their phase.
  Don't over-engineer; clean and readable beats clever.

## Data  (durable)
- Bank Account Fraud (BAF) `Base.csv` from Kaggle
  (`sgpjesus/bank-account-fraud-dataset-neurips-2022`), 1,000,000 rows, ~1.1% fraud.
  `Base` is the least adversarial of the six BAF variants — relevant when sanity-checking
  metrics against published baselines.
- Acquire via `ml/src/data.py`; needs `KAGGLE_USERNAME` + `KAGGLE_KEY` in a `.env` at repo
  root (the kaggle client ignores any other variable name, e.g. `KAGGLE_API_TOKEN`).
- `.env`, `data/`, `models/`, and `personal_notes.txt` are gitignored — never commit them.
- Missing values are encoded as negatives; `preprocessing.clean()` handles them. The
  temporal split (train months 0–5, val 6, test 7) is the only valid split — never random.
