# Phase 1 — Plain-English Summary & Vocabulary

A simplified recap of Phase 1 (ML baseline + cost-sensitive evaluation) plus the vocabulary
worth knowing. For the numbers, see [baseline_metrics.md](baseline_metrics.md).

## What we did, step by step

- We set up a clean project skeleton (folders, virtual environment, pinned dependencies,
  gitignore) so later phases can reuse the code.
- We picked **LightGBM** as our main model library (good at messy tabular data with categories).
- We worked out how to download the data from Kaggle, and **debugged the Kaggle login** — the
  tool needs `KAGGLE_USERNAME` + `KAGGLE_KEY`, not a single token.
- We downloaded the Bank Account Fraud dataset (1 million rows) and **recorded its exact
  fingerprint** (a SHA-256 hash) so we always know which version we used.
- We explored the data (EDA) and learned the key facts: only **1.1% of cases are fraud**, the
  data spans 8 months, "missing" values are hidden as negative numbers, and one column was
  useless (always zero).
- We discovered the fraud rate **drifts upward over time** — this shaped everything after.
- We made a **notebook** version of the exploration to revisit later.
- We cleaned the data: dropped the dead column, turned hidden "missing" values into proper
  missing markers plus flags.
- We split the data **by time** (early months train, later months test) instead of randomly —
  random would let the model cheat by seeing the future.
- We built **Logistic Regression** to set a baseline "floor" to beat.
- We built a **Random Forest** — barely better than the floor.
- We built **LightGBM** — clearly the best and the fastest; we picked it as the winner.
- For every model we used the same honest scorecard (**PR-AUC**, recall, etc.), not accuracy —
  accuracy lies when fraud is rare.
- Then the main event: we assigned **dollar costs** to mistakes (missing a fraud = $500, false
  alarm = $50).
- We found the decision **threshold that costs the least money**, instead of 0.5 or the F1-best.
- We showed the default 0.5 threshold is **worse than doing nothing**, and the best threshold
  **changes as the fraud rate changes**.
- We did it all **leakage-free**: chose the threshold on validation data, tested once on truly
  unseen data.
- We **saved the final model** plus a metadata file (costs, threshold, data hash, versions) for
  full reproducibility.
- We wrote up everything in **reports** and updated the README.

## Vocabulary to know

Try to explain each back in your own words; the note in parentheses ties it to our project.

### The problem
- **Class imbalance** — one class is far rarer than the other (fraud ≈ 1%).
- **Base rate / prevalence** — how common the positive class is (the 1.1% fraud rate).
- **Concept drift** — patterns/rates change over time (fraud rose month to month).
- **Synthetic data** — artificially generated data mimicking real data (BAF is synthetic).

### Data prep
- **EDA (Exploratory Data Analysis)** — the first investigative look at the data.
- **Cardinality** — how many distinct values a column has (low = few, high = many).
- **Categorical vs numeric feature** — labels/categories vs numbers.
- **One-hot encoding** — turning a category column into several 0/1 columns.
- **Missing value / sentinel value** — absent data; a sentinel is a placeholder (e.g. `-1`).
- **Imputation** — filling in missing values (we used the median).
- **Feature** — an input column. **Target / label** — what we predict (`fraud_bool`).

### Splitting & honesty
- **Train / validation / test split** — data for learning / tuning / final unbiased scoring.
- **Temporal split** — splitting by time instead of randomly.
- **Data leakage** — model gets info it wouldn't have in real life, inflating scores.
- **Reproducibility** — getting the exact same results again (seeds, pinned versions, hash).
- **Random seed** — a fixed number making "random" steps repeatable (`seed=42`).

### Models
- **Logistic Regression** — a simple linear yes/no model; our baseline.
- **Random Forest** — many decision trees averaged together.
- **Gradient boosting / LightGBM** — trees built one after another, each fixing prior mistakes.
- **Class weighting (`balanced`)** — telling the model to care more about the rare class.
- **Hyperparameters** — settings chosen before training (e.g. number of trees).
- **Pipeline** — preprocessing + model bundled so steps fit only on train (no leakage).
- **Overfitting** — memorizing training data and failing on new data.

### Scoring / metrics
- **Confusion matrix** — table of TP / FP / FN / TN counts.
- **TP / FP / FN / TN** — the four outcomes (FN = a missed fraud).
- **Precision** — of what we flagged, how much was really fraud.
- **Recall** — of all real fraud, how much we caught.
- **F1 score** — the balance (harmonic mean) of precision and recall.
- **Accuracy** — % of all predictions correct (misleading under imbalance).
- **ROC-AUC** — ranking quality; looks deceptively good here.
- **PR-AUC (average precision)** — ranking quality focused on the rare class; our main metric.
- **Probability score** — the model's 0–1 "how fraud-like" output.

### The decision layer (the differentiator)
- **Decision threshold** — the cutoff turning a score into yes/no.
- **Cost matrix** — the dollar cost of each kind of mistake.
- **Cost-sensitive / expected-cost minimization** — picking the threshold that loses least money.
- **Calibration** — whether a "0.10" really means a 10% chance (ours isn't calibrated; fine here).
- **Model artifact** — the saved, loadable trained model file.
