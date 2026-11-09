import json

import numpy as np
import pandas as pd
import pytest

from logad.evaluation import evaluate_scores
from logad.evaluation.metrics import (
    Counts,
    best_f1_threshold,
    bootstrap_prf,
    counts_at,
    evaluate_model,
    prf,
)


def test_counts_at_alerts_on_score_at_or_above_threshold():
    counts = counts_at([1, 1, 0, 0, 1], [0.9, 0.5, 0.5, 0.1, 0.2], threshold=0.5)
    assert counts == Counts(tp=2, fp=1, fn=1, tn=1)
    assert counts.alerts == 3


def test_prf_values_and_undefined_cases():
    precision, recall, f1 = prf(tp=8, fp=2, fn=8)
    assert (float(precision), float(recall)) == (0.8, 0.5)
    assert float(f1) == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert [float(v) for v in prf(0, 0, 5)] == [0.0, 0.0, 0.0]  # no alerts → 0, not a crash


def test_best_f1_threshold_separates_perfectly_separable_scores():
    threshold, f1 = best_f1_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9])
    assert f1 == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.8


def test_best_f1_threshold_beats_other_thresholds():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.05
    scores = rng.normal(size=2000) + 2.5 * y
    threshold, best = best_f1_threshold(y, scores)
    for t in np.quantile(scores, [0.5, 0.8, 0.9, 0.99]):
        c = counts_at(y, scores, t)
        assert float(prf(c.tp, c.fp, c.fn)[2]) <= best + 1e-12


def test_bootstrap_interval_contains_point_estimate_and_shrinks_with_more_data():
    small = Counts(tp=40, fp=10, fn=20, tn=930)
    large = Counts(tp=4000, fp=1000, fn=2000, tn=93000)
    f1 = float(prf(small.tp, small.fp, small.fn)[2])
    lo, hi = bootstrap_prf(small)["f1"]
    assert lo <= f1 <= hi
    lo_l, hi_l = bootstrap_prf(large)["f1"]
    assert (hi_l - lo_l) < (hi - lo)


def test_bootstrap_is_reproducible():
    counts = Counts(tp=40, fp=10, fn=20, tn=930)
    assert bootstrap_prf(counts, seed=1) == bootstrap_prf(counts, seed=1)


def test_evaluate_model_uses_validation_threshold_on_test():
    y_val, s_val = [0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]
    y_test, s_test = [0, 0, 1, 1, 0], [0.1, 0.8, 0.75, 0.3, 0.2]
    result = evaluate_model("m", y_val, s_val, y_test, s_test, n_resamples=200)
    assert result.threshold == pytest.approx(0.7)
    assert (result.tp, result.fp, result.fn) == (1, 1, 1)  # the test set is harder
    assert result.precision == pytest.approx(0.5)


def make_scores(tmp_path, labelled=True):
    rng = np.random.default_rng(0)
    n = 600
    split = ["train"] * 360 + ["validation"] * 120 + ["test"] * 120
    label = (rng.random(n) < 0.1).astype(int)
    df = pd.DataFrame(
        {
            "group_id": [f"blk_{i}" for i in range(n)],
            "first_seen": pd.date_range("2008-11-09", periods=n, freq="min"),
            "split": split,
            "label": pd.array(label if labelled else [None] * n, dtype="Int8"),
            "score_good": rng.normal(size=n) + 3 * label,
            "score_random": rng.normal(size=n),
        }
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    df.to_parquet(artifacts / "scores.parquet", index=False)
    return {
        "dataset": "test",
        "seed": 0,
        "paths": {"artifacts_dir": str(artifacts), "reports_dir": str(tmp_path / "reports")},
    }


def test_evaluate_scores_end_to_end(tmp_path):
    config = make_scores(tmp_path)
    results = evaluate_scores(config, n_resamples=200).set_index("model")

    assert results.loc["good", "f1"] > results.loc["random", "f1"]
    assert results.loc["good", "roc_auc"] > 0.9
    thresholds = json.loads((tmp_path / "artifacts" / "thresholds.json").read_text())
    assert set(thresholds) == {"good", "random"}
    reports = tmp_path / "reports"
    assert (reports / "pr_curves.png").stat().st_size > 0
    assert (reports / "score_distributions.png").stat().st_size > 0
    assert "| F1 |" in (reports / "results.md").read_text(encoding="utf-8")


def test_evaluate_scores_refuses_unlabelled_data(tmp_path):
    config = make_scores(tmp_path, labelled=False)
    with pytest.raises(ValueError, match="no labels"):
        evaluate_scores(config)


def test_evaluate_scores_without_training_explains_what_to_run(tmp_path):
    config = {"paths": {"artifacts_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="logad train"):
        evaluate_scores(config)
