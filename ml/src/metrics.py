"""Evaluation metrics for imbalanced fraud detection.

Headline numbers are **threshold-independent** (PR-AUC, ROC-AUC) so models are
compared fairly before we pick a decision threshold (that is Step 8). The at-threshold
metrics (precision/recall/F1/confusion matrix) are reported at a given threshold for
illustration; PR-AUC is the primary model-selection metric here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvalResult:
    model: str
    threshold: float
    prevalence: float  # base fraud rate in this eval set (= random-model PR-AUC)
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        lift = self.pr_auc / self.prevalence if self.prevalence else float("nan")
        return (
            f"{self.model}  (threshold={self.threshold:.3f})\n"
            f"  PR-AUC   : {self.pr_auc:.4f}   (vs {self.prevalence:.4f} random  ->  {lift:.1f}x lift)\n"
            f"  ROC-AUC  : {self.roc_auc:.4f}\n"
            f"  Precision: {self.precision:.4f}   Recall: {self.recall:.4f}   F1: {self.f1:.4f}\n"
            f"  Confusion: TP={self.tp:>5}  FP={self.fp:>6}  FN={self.fn:>5}  TN={self.tn:>7}"
        )


def evaluate(y_true, scores, threshold: float = 0.5, model: str = "model") -> EvalResult:
    """Compute PR-AUC, ROC-AUC, and at-threshold precision/recall/F1/confusion.

    ``scores`` are predicted probabilities (or any ranking score) for the positive class.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EvalResult(
        model=model,
        threshold=threshold,
        prevalence=float(y_true.mean()),
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


# --- cost-sensitive thresholding (Step 8) -----------------------------------


def cost_at_thresholds(y_true, scores, c_fn: float, c_fp: float, thresholds=None):
    """Total dollar cost at each candidate threshold, computed exactly.

    Cost = FN*c_fn + FP*c_fp (TP/TN cost 0). Uses class-conditional sorted scores +
    searchsorted, so it is O(n log n) rather than a quadratic sweep. Returns
    (thresholds, cost, tp, fp, fn, fnr, fpr); fnr/fpr are prevalence-independent and
    drive the base-rate sensitivity analysis.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = np.sort(scores[y_true == 1])
    neg = np.sort(scores[y_true == 0])
    P, N = len(pos), len(neg)
    if thresholds is None:
        thresholds = np.unique(scores)
    thresholds = np.asarray(thresholds)

    # count of scores >= t  ==  len - first index >= t
    tp = P - np.searchsorted(pos, thresholds, side="left")
    fp = N - np.searchsorted(neg, thresholds, side="left")
    fn = P - tp
    cost = fn * c_fn + fp * c_fp
    fnr = fn / P
    fpr = fp / N
    return thresholds, cost, tp, fp, fn, fnr, fpr


def cost_optimal_threshold(y_true, scores, c_fn: float, c_fp: float):
    """Threshold that minimizes total expected dollar cost. Returns a dict."""
    thr, cost, tp, fp, fn, _, _ = cost_at_thresholds(y_true, scores, c_fn, c_fp)
    i = int(np.argmin(cost))
    return {
        "threshold": float(thr[i]),
        "cost": float(cost[i]),
        "tp": int(tp[i]),
        "fp": int(fp[i]),
        "fn": int(fn[i]),
    }


def f1_optimal_threshold(y_true, scores):
    """Threshold that maximizes F1 — the cost-blind benchmark we compare against."""
    thr, _, tp, fp, fn, _, _ = cost_at_thresholds(y_true, scores, 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = tp / (tp + fn)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    i = int(np.argmax(f1))
    return {"threshold": float(thr[i]), "f1": float(f1[i])}


def optimal_threshold_vs_base_rate(y_true, scores, c_fn, c_fp, base_rates):
    """How the cost-optimal threshold shifts as fraud prevalence changes.

    Uses prevalence-independent FNR(t)/FPR(t) from the test scores, then for each
    target prevalence pi computes per-instance expected cost
    pi*FNR*c_fn + (1-pi)*FPR*c_fp and takes the argmin threshold. No resampling noise.
    """
    thr, _, _, _, _, fnr, fpr = cost_at_thresholds(y_true, scores, c_fn, c_fp)
    rows = []
    for pi in base_rates:
        cost = pi * fnr * c_fn + (1 - pi) * fpr * c_fp
        i = int(np.argmin(cost))
        rows.append({"base_rate": float(pi), "threshold": float(thr[i]),
                     "cost_per_case": float(cost[i])})
    return rows
