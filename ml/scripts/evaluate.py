"""Evaluate the saved baseline on the sealed test set (month 7).

Thin entrypoint — all logic lives in ml/src/. Run from the repo root:
    python -m ml.scripts.evaluate
"""
from ml.src import config, metrics
from ml.src import models
from ml.src import preprocessing as pp


def main() -> None:
    pipe, threshold = models.load_baseline()

    _, _, test = pp.load_processed()
    Xte, yte = pp.split_xy(test)
    scores = pipe.predict_proba(Xte)[:, 1]

    # Honest report: threshold was chosen on val, applied unchanged to test.
    res = metrics.evaluate(yte, scores, threshold, model="LightGBM (test/month7)")
    print(res.summary())

    fp = res.fp
    fn = res.fn
    total_cost = fn * config.COST_FN + fp * config.COST_FP
    print(
        f"\nAt threshold {threshold:.3f}  (FN=${config.COST_FN:.0f}, FP=${config.COST_FP:.0f}):"
    )
    print(f"  total cost on test = ${total_cost:,.0f}  ({fn} FN x ${config.COST_FN:.0f} + {fp} FP x ${config.COST_FP:.0f})")


if __name__ == "__main__":
    main()
