import numpy as np
import pandas as pd
import pytest

from logad.models import MODELS, build_models, quick_check, train_models
from logad.models.base import AnomalyDetector
from logad.models.baseline import RareEventDetector
from logad.models.iforest import IsolationForestDetector
from logad.models.pca import PCADetector


def normal_blocks(n=300, seed=0):
    """Synthetic 'normal' blocks: events 0-2 always together in fixed proportions
    (like receive → respond → store), event 3 never happens."""
    rng = np.random.default_rng(seed)
    k = rng.integers(1, 4, size=n)
    return np.column_stack([3 * k, 3 * k, 3 * k, np.zeros(n)]).astype(float)


NORMAL = np.array([[6.0, 6.0, 6.0, 0.0]])
ODD_MIX = np.array([[9.0, 0.0, 1.0, 0.0]])  # usual events, unusual proportions
RARE_EVENT = np.array([[6.0, 6.0, 6.0, 1.0]])  # contains an event never seen in training


@pytest.mark.parametrize("model_cls", [RareEventDetector, PCADetector])
def test_detectors_flag_event_never_seen_in_training(model_cls):
    detector = model_cls(seed=0).fit(normal_blocks())
    assert detector.score(RARE_EVENT)[0] > detector.score(NORMAL)[0]


def test_isolation_forest_is_blind_to_events_never_seen_in_training():
    # Known limitation: trees can only split on features that VARY in the training data.
    # Event 3 is always 0 in training, so no tree ever looks at it.
    detector = IsolationForestDetector(seed=0).fit(normal_blocks())
    assert detector.score(RARE_EVENT)[0] == detector.score(NORMAL)[0]


@pytest.mark.parametrize("model_cls", [PCADetector, IsolationForestDetector])
def test_ml_detectors_catch_unusual_proportions(model_cls):
    # The rule baseline can't see this (no rare event), the ML models should.
    detector = model_cls(seed=0).fit(normal_blocks())
    assert detector.score(ODD_MIX)[0] > detector.score(NORMAL)[0]


def test_baseline_rarity_and_explanation():
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]])
    detector = RareEventDetector().fit(X)
    # event 0 in all blocks → rarity 0; event 1 in 1 of 4 → -log10(0.25); event 2 never seen
    assert detector.rarity_[0] == pytest.approx(0.0)
    assert detector.rarity_[1] == pytest.approx(-np.log10(0.25))
    assert detector.rarity_[2] == pytest.approx(-np.log10(1 / 5))
    assert detector.score([[2, 0, 0]])[0] == pytest.approx(0.0)
    assert detector.rarest_event([[1, 1, 1]])[0] == 2


def test_predict_uses_threshold():
    detector = RareEventDetector().fit(normal_blocks())
    flags = detector.predict(np.vstack([NORMAL, RARE_EVENT]), threshold=1.0)
    assert flags.tolist() == [0, 1]


def test_pca_keeps_enough_components():
    detector = PCADetector(variance_kept=0.95, transform="event_count").fit(normal_blocks())
    assert detector.n_components == 1  # the synthetic data has one real pattern


def test_same_seed_same_scores():
    X = normal_blocks()
    a = IsolationForestDetector(seed=7).fit(X).score(X)
    b = IsolationForestDetector(seed=7).fit(X).score(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("model_cls", [RareEventDetector, PCADetector, IsolationForestDetector])
def test_save_and_load(tmp_path, model_cls):
    detector = model_cls(seed=0).fit(normal_blocks())
    path = tmp_path / "model.joblib"
    detector.save(path)
    loaded = AnomalyDetector.load(path)
    assert np.allclose(loaded.score(ODD_MIX), detector.score(ODD_MIX))


def test_build_models_uses_per_model_and_global_settings():
    config = {
        "seed": 3,
        "features": {"type": "event_count", "scale": False},
        "models": {"baseline": {}, "pca": {"transform": "tfidf"}, "isolation_forest": None},
    }
    detectors = build_models(config)
    assert list(detectors) == ["baseline", "pca", "isolation_forest"]
    assert detectors["pca"].transform == "tfidf"  # per-model override
    assert detectors["isolation_forest"].transform == "event_count"  # global default
    assert detectors["isolation_forest"].seed == 3


def test_build_models_rejects_unknown_or_empty():
    with pytest.raises(ValueError, match="Unknown model"):
        build_models({"models": {"lstm": {}}})
    with pytest.raises(ValueError, match="no models"):
        build_models({"models": {}})


def test_registry_names_match_classes():
    for name, cls in MODELS.items():
        assert cls.name == name


def test_quick_check():
    labels = pd.Series([0, 0, 1, 1], dtype="Int8")
    roc, pr = quick_check(labels, [0.1, 0.2, 0.8, 0.9])
    assert roc == pytest.approx(1.0) and pr == pytest.approx(1.0)
    assert quick_check(pd.Series([pd.NA] * 4, dtype="Int8"), [1, 2, 3, 4]) == (None, None)


def test_train_models_end_to_end(tmp_path):
    X = np.vstack([normal_blocks(80), ODD_MIX, RARE_EVENT, normal_blocks(20, seed=1), RARE_EVENT])
    n = len(X)
    df = pd.DataFrame(X.astype("int32"), columns=["E1", "E2", "E3", "E4"])
    df.insert(0, "group_id", [f"blk_{i}" for i in range(n)])
    df.insert(1, "first_seen", pd.date_range("2008-11-09", periods=n, freq="min"))
    df.insert(2, "split", ["train"] * 82 + ["validation"] * (n - 82))
    labels = [0] * n
    labels[80] = labels[81] = labels[-1] = 1
    df.insert(3, "label", pd.array(labels, dtype="Int8"))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    df.to_parquet(artifacts / "features.parquet", index=False)
    config = {
        "seed": 0,
        "paths": {"artifacts_dir": str(artifacts)},
        "models": {"baseline": {}, "pca": {"transform": "event_count"}},
    }

    results = train_models(config)

    assert [r.model for r in results] == ["baseline", "pca"]
    assert results[0].val_roc_auc == pytest.approx(1.0)  # the rare event is caught
    scores = pd.read_parquet(artifacts / "scores.parquet")
    assert len(scores) == n
    assert {"score_baseline", "score_pca"} <= set(scores.columns)
    assert (artifacts / "models" / "pca.joblib").is_file()
