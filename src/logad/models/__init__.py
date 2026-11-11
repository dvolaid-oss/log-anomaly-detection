"""Anomaly detectors sharing one interface: fit(X), score(X), predict(X, threshold).

`train_models(config)` fits every model listed in the config's `models` section on the
TRAIN split (no labels), saves each fitted model, and scores every block in every split
into `scores.parquet` for evaluation (Stage 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from logad.features import load_features
from logad.models.autoencoder import AutoencoderDetector  # imports torch only when used
from logad.models.base import AnomalyDetector
from logad.models.baseline import RareEventDetector
from logad.models.iforest import IsolationForestDetector
from logad.models.pca import PCADetector

MODELS: dict[str, type[AnomalyDetector]] = {
    "baseline": RareEventDetector,
    "pca": PCADetector,
    "isolation_forest": IsolationForestDetector,
    "autoencoder": AutoencoderDetector,  # optional: needs requirements-torch.txt
}
MODELS_DIR = "models"
SCORES_FILE = "scores.parquet"


@dataclass
class TrainResult:
    model: str
    fit_seconds: float
    val_roc_auc: float | None = None  # quick sanity check; full evaluation is Stage 5
    val_pr_auc: float | None = None
    val_anomaly_share: float | None = None  # PR-AUC a random guesser would get


def build_models(config: dict[str, Any]) -> dict[str, AnomalyDetector]:
    """Create (unfitted) detectors from the config.

    Each `models.<name>` entry holds that model's settings. `transform` / `scale` default to
    the global `features` section, so a model can override them (PCA uses TF-IDF).
    """
    features = config.get("features", {})
    detectors = {}
    for name, settings in (config.get("models") or {}).items():
        if name not in MODELS:
            raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODELS)}")
        settings = dict(settings or {})
        settings.setdefault("transform", features.get("type", "event_count"))
        settings.setdefault("scale", features.get("scale", False))
        detectors[name] = MODELS[name](seed=config.get("seed", 42), **settings)
    if not detectors:
        raise ValueError("Config has no models to train (the 'models' section is empty).")
    return detectors


def quick_check(labels: pd.Series, scores) -> tuple[float | None, float | None]:
    """ROC-AUC and PR-AUC: how well scores rank anomalies above normal blocks.

    Threshold-free. Labels are only READ here, to report; training never sees them.
    Returns (None, None) if labels are missing or only one class is present.
    """
    known = labels.notna()
    y = labels[known].astype(int)
    if y.nunique() < 2:
        return None, None
    s = pd.Series(scores, index=labels.index)[known]
    return roc_auc_score(y, s), average_precision_score(y, s)


def train_models(config: dict[str, Any]) -> list[TrainResult]:
    """Fit all configured models on the train split, save them, and score every block."""
    df, events = load_features(config)
    artifacts = Path(config["paths"]["artifacts_dir"])
    X_all = df[events].to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "validation").to_numpy()

    scores = df[["group_id", "first_seen", "split", "label"]].copy()
    results = []
    for name, detector in build_models(config).items():
        start = time.perf_counter()
        detector.fit(X_all[is_train])  # ← only training blocks, no labels
        fit_seconds = time.perf_counter() - start

        scores[f"score_{name}"] = detector.score(X_all)
        detector.save(artifacts / MODELS_DIR / f"{name}.joblib")
        val_labels = df.loc[is_val, "label"]
        roc, pr = quick_check(val_labels, scores.loc[is_val, f"score_{name}"])
        share = float(val_labels.mean()) if val_labels.notna().any() else None
        results.append(TrainResult(name, fit_seconds, roc, pr, share))

    part = artifacts / (SCORES_FILE + ".part")
    scores.to_parquet(part, index=False)
    part.replace(artifacts / SCORES_FILE)
    return results
