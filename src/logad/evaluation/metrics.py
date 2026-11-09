"""Metrics for anomaly detection: thresholds, precision/recall/F1, confidence intervals.

Vocabulary (anomaly = positive class):
    TP  caught anomaly     (alert, really anomalous)
    FP  false alarm        (alert, actually normal)
    FN  missed anomaly     (no alert, really anomalous)
    precision = TP / (TP + FP)   "of all alerts, how many were real?"
    recall    = TP / (TP + FN)   "of all anomalies, how many did we catch?"
    F1        = harmonic mean of the two: high only when BOTH are high
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass
class Counts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def alerts(self) -> int:
        return self.tp + self.fp


def counts_at(y_true, scores, threshold: float) -> Counts:
    """Confusion counts when alerting on score >= threshold."""
    y = np.asarray(y_true, dtype=bool)
    alert = np.asarray(scores) >= threshold
    return Counts(
        tp=int((alert & y).sum()),
        fp=int((alert & ~y).sum()),
        fn=int((~alert & y).sum()),
        tn=int((~alert & ~y).sum()),
    )


def prf(tp, fp, fn):
    """Precision, recall, F1 from counts. Works on numbers or numpy arrays; 0 when undefined."""
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """The threshold with the highest F1, and that F1 (alert when score >= threshold).

    Used on the VALIDATION split only: the one place labels influence the system.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold (it means "alert on nothing").
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def bootstrap_prf(counts: Counts, n_resamples: int = 1000, seed: int = 42) -> dict[str, tuple]:
    """95% confidence intervals for precision, recall and F1.

    A bootstrap redraws the test set with replacement many times and recomputes the metric,
    to see how much it would wobble on a slightly different test set. Each block falls into
    exactly one of TP/FP/FN/TN, so redrawing n blocks is the same as drawing the four counts
    from a multinomial distribution: identical result, without copying 100k rows 1,000 times.
    """
    n = counts.tp + counts.fp + counts.fn + counts.tn
    shares = np.array([counts.tp, counts.fp, counts.fn, counts.tn]) / n
    draws = np.random.default_rng(seed).multinomial(n, shares, size=n_resamples)
    precision, recall, f1 = prf(draws[:, 0], draws[:, 1], draws[:, 2])
    return {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in [("precision", precision), ("recall", recall), ("f1", f1)]
    }


@dataclass
class TestResult:
    model: str
    threshold: float
    validation_f1: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    alerts: int
    roc_auc: float
    pr_auc: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    f1_ci: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_model(
    model: str, y_val, s_val, y_test, s_test, n_resamples: int = 1000, seed: int = 42
) -> TestResult:
    """Pick the threshold on validation, then measure on test (once)."""
    threshold, validation_f1 = best_f1_threshold(y_val, s_val)
    counts = counts_at(y_test, s_test, threshold)
    precision, recall, f1 = (float(v) for v in prf(counts.tp, counts.fp, counts.fn))
    ci = bootstrap_prf(counts, n_resamples, seed)
    return TestResult(
        model=model,
        threshold=threshold,
        validation_f1=validation_f1,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        alerts=counts.alerts,
        roc_auc=float(roc_auc_score(y_test, s_test)),
        pr_auc=float(average_precision_score(y_test, s_test)),
        precision_ci=ci["precision"],
        recall_ci=ci["recall"],
        f1_ci=ci["f1"],
    )
