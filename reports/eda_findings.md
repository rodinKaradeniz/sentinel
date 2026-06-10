# EDA Findings — BAF Base.csv (Phase 1, Step 3)

Source: `data/raw/Base.csv`, SHA-256 `7bf10a37…dcf9b809` (see dataset_version.md).
1,000,000 rows × 32 columns. No explicit NaNs.

## Class imbalance
- Fraud rate **1.10%** (11,029 positives). ~89:1. → optimize PR-AUC / recall, not accuracy.

## Temporal structure (`month` 0–7)
Row counts decline over time; fraud rate drifts **upward** in later months:

| month | rows  | fraud rate |
|------:|------:|-----------:|
| 0     | 132k  | 1.13%      |
| 1     | 128k  | 0.94%      |
| 2     | 137k  | 0.88%      |
| 3     | 151k  | 0.92%      |
| 4     | 128k  | 1.14%      |
| 5     | 119k  | 1.18%      |
| 6     | 108k  | 1.34%      |
| 7     | 97k   | 1.48%      |

→ Concept drift is real. Motivates a **temporal split** (train early, test late) and feeds the
Step 8 analysis (optimal threshold shifts with base rate).

## Categoricals (all low cardinality → one-hot is safe)
`payment_type` (5), `employment_status` (7), `housing_status` (7), `source` (2), `device_os` (5).

## Missingness via negative sentinels (no NaNs in file)
Treat in Step 4 by group:

- **`-1` = missing** → add missing flag, map -1→NaN (LightGBM handles NaN natively):
  - `prev_address_months_count` (71.3% missing)
  - `bank_months_count` (25.4%)
  - `current_address_months_count` (0.4%)
  - `session_length_in_minutes` (0.2%)
  - `device_distinct_emails_8w` (0.04%)
- **`intended_balcon_amount`**: 74.3% negative, continuous (min -15.5) → own missing flag.
- **Keep as real (do NOT treat as missing):** `credit_risk_score` (1.4% negative — a score),
  `velocity_6h` (44 negative rows — rare noise; flag only).

## Leakage
Clean. Max |corr(feature, target)| = 0.071 (`credit_risk_score`). No post-event columns.

## Drop
- `device_fraud_count`: constant (all 0) → zero information.
