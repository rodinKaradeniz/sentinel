# Model Comparison — Baseline Selection (Phase 1, Steps 5–7)

**Goal:** pick the baseline model on *validation* (month 6) before any threshold or cost
decision. All three models were fit on **train (months 0–5)** with no access to validation
or test, so the comparison is apples-to-apples. Final performance is reported separately on
the sealed **test (month 7)** set in [baseline_metrics.md](baseline_metrics.md).

## Why PR-AUC is the selection metric
Fraud rate is ~1.1% (≈89:1). Accuracy is meaningless (predict "never fraud" → 98.9% accurate,
catches nothing). ROC-AUC is inflated by the 89× more numerous legit cases in its
false-positive-rate denominator. **PR-AUC** only involves frauds and flagged cases, so it
stays honest about the thing we care about. Random-model PR-AUC ≈ base rate, evaluated on
this set: **month 6 = 0.0134**, which is the reference for "lift" *here*. (The final test
report uses month 7, prevalence 0.0147 — a different floor for a different set; see
[baseline_metrics.md](baseline_metrics.md).)

## Results (validation / month 6)

| Model                | **PR-AUC** | ROC-AUC | Train time | Preprocessing                  |
|----------------------|-----------:|--------:|-----------:|--------------------------------|
| Logistic Regression  | 0.1554     | 0.8822  | 8s         | median impute + scale + one-hot|
| Random Forest        | 0.1594     | 0.8827  | 64s        | median impute + one-hot        |
| **LightGBM**         | **0.1794** | 0.8940  | 7s         | native categorical + NaN       |

All used class weighting (`balanced` / `balanced_subsample`) to counter the imbalance.

## Verdict: LightGBM
- **Best PR-AUC**: +0.024 over RF (~+13% relative) and over LR. LR and RF landed within 0.004
  of each other (the signal they capture is near-linear); LightGBM's gain comes from
  sequential boosting capturing **non-linear feature interactions** the others missed.
- **Fastest (7s)** *and* **simplest prep** — native categorical/NaN handling means no one-hot
  and no imputation. Best accuracy, least preprocessing, least time.
- ROC-AUC barely moved (0.882 → 0.894) while PR-AUC rose ~15% — a concrete reminder that
  ROC-AUC was nearly blind to the improvement that matters here.

## Notes / honesty
- All three plateau in the **0.15–0.18 PR-AUC** band → this is a deliberately hard, realistically
  imbalanced dataset with no magic feature; ~0.18 looks like the untuned-baseline ceiling.
- No hyperparameter tuning and no early stopping on validation were used, to keep the
  comparison fair. The Phase-1 differentiator is the cost-sensitive threshold analysis (Step 8),
  not squeezing another 0.01 of PR-AUC.
