# Baseline Metrics — Sentinel Phase 1

Final results for the Phase-1 fraud baseline on the **Bank Account Fraud (BAF) `Base.csv`**
dataset. This is the consolidated report; see also:
- [model_comparison.md](model_comparison.md) — how LightGBM was chosen (validation).
- [cost_sensitive_analysis.md](cost_sensitive_analysis.md) — the cost-optimal threshold work.
- [dataset_version.md](dataset_version.md) — exact data version / hash.
- [eda_findings.md](eda_findings.md) — EDA.

## Setup (no leakage)
- **Temporal split:** train = months 0–5, val = month 6, test = month 7. Earlier → later,
  never random — a random split would train on the future and inflate metrics.
- **Model:** LightGBM (native categorical + NaN handling), `class_weight="balanced"`.
  Untuned sensible defaults. Trained on **train** only.
- **Threshold:** cost-optimal cut selected on **val** (month 6), then applied unchanged to
  **test** (month 7). Test was sealed until final evaluation.
- **Cost matrix:** FN (approve a fraud) = $500, FP (block a legit) = $50 → 10:1.
- Reproducibility: `seed=42`; dataset SHA-256 and library versions recorded in
  `models/lightgbm_baseline.meta.json`.

## Model selection (validation / month 6)

| Model | **PR-AUC** | ROC-AUC | Train time |
|---|---:|---:|---:|
| Logistic Regression | 0.1554 | 0.8822 | 8s |
| Random Forest | 0.1594 | 0.8827 | 64s |
| **LightGBM** | **0.1794** | 0.8940 | 7s |

PR-AUC is the selection metric (fraud ≈1.1%, so accuracy/ROC-AUC mislead). LightGBM wins on
accuracy, speed, and simplest preprocessing. Random-model PR-AUC ≈ base rate.

## Final performance (sealed test / month 7)

LightGBM, threshold **0.864** (chosen on val):

```
PR-AUC   : 0.2099   (vs 0.0147 random  ->  14.2x lift)
ROC-AUC  : 0.8962
Precision: 0.218    Recall: 0.406    F1: 0.284
Confusion: TP=580  FP=2081  FN=848  TN=93334
```

## Cost outcome (test / month 7)

| Policy | Threshold | Total cost | Note |
|---|---:|---:|---|
| Approve everything | — | $714,000 | miss all fraud |
| Default 0.5 | 0.500 | $842,700 | **worse than doing nothing** |
| F1-optimal (oracle) | 0.861 | $525,100 | cost-blind |
| **Deployed (val-chosen)** | **0.864** | **$528,050** | leakage-free |
| Cost-optimal (oracle) | 0.826 | $521,650 | upper bound |

- The deployed, leakage-free threshold costs **$528,050** vs the test oracle's $521,650 — a
  $6,400 gap driven by val→test base-rate drift (1.34% → 1.48%).
- It saves **26% vs approving everything** and **37% vs the default 0.5 threshold**.
- See [cost_sensitive_analysis.md](cost_sensitive_analysis.md) for the cost-vs-threshold
  curve, the cost-ratio sensitivity (F1 is only coincidentally near-optimal at 10:1), and the
  base-rate shift analysis.

## Key takeaways
1. **Imbalance makes accuracy/ROC-AUC misleading**; PR-AUC and recall are the honest metrics.
2. **The threshold is a business decision, not 0.5.** Minimizing expected cost ≠ maximizing
   F1; the default 0.5 is actively harmful here.
3. **The optimal threshold tracks the base rate** — concept drift means it must be re-tuned,
   not frozen (a monitoring hook for later phases).

## Artifact
- `models/lightgbm_baseline.joblib` — `{pipeline, threshold}`.
- `models/lightgbm_baseline.meta.json` — costs, threshold, seed, dataset hash, versions.

## Reproduce
```bash
python -m ml.scripts.train      # train on train, pick threshold on val, save artifact
python -m ml.scripts.evaluate   # evaluate saved artifact on sealed test
```
