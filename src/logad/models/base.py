"""The interface every anomaly detector shares.

    detector.fit(X_train)          # learn what "normal" looks like (NO labels)
    scores = detector.score(X)     # higher score = more anomalous
    flags = detector.predict(X, threshold)   # 1 = anomaly, 0 = normal

Each detector owns its feature transformer (plain counts, TF-IDF, scaling), fitted together
with the model on the training data. Code that uses detectors never needs to know which
model or transform is inside, so models are swappable and comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from logad.features.transform import make_transformer


class AnomalyDetector(ABC):
    """Base class: subclasses implement `_fit` and `_score` on already-transformed data."""

    name = "base"

    def __init__(self, transform: str = "event_count", scale: bool = False, seed: int = 42):
        self.transform = transform
        self.scale = scale
        self.seed = seed
        self.transformer = make_transformer(transform, scale)

    def fit(self, X) -> AnomalyDetector:
        """Fit transformer + model on training data. Labels are never passed in."""
        self._fit(self.transformer.fit_transform(X))
        return self

    def score(self, X) -> np.ndarray:
        """Anomaly score per row: higher = more anomalous."""
        return np.asarray(self._score(self.transformer.transform(X)), dtype=float)

    def predict(self, X, threshold: float) -> np.ndarray:
        """1 where score >= `threshold`, else 0 (thresholds are chosen by `logad evaluate`)."""
        return (self.score(X) >= threshold).astype(int)

    def reconstruction(self, X) -> tuple[np.ndarray, np.ndarray]:
        """For models that REBUILD blocks (PCA, autoencoder): per row, the residual in the
        model's own feature space, and the event counts the model expected to see.
        Used to explain alerts ("expected ~3x E3, found 0")."""
        Xt = self.transformer.transform(X)
        rebuilt = self._rebuild(Xt)
        return Xt - rebuilt, np.maximum(self._to_counts(rebuilt), 0)

    def _rebuild(self, Xt: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not rebuild blocks")

    @property
    def rebuilds(self) -> bool:
        return type(self)._rebuild is not AnomalyDetector._rebuild

    def _to_counts(self, Xt: np.ndarray) -> np.ndarray:
        """Undo the feature transform (scaling, then TF-IDF) to get back to event counts."""
        steps = self.transformer.named_steps
        if "scale" in steps:
            Xt = steps["scale"].inverse_transform(Xt)
        if "tfidf" in steps:
            Xt = Xt / steps["tfidf"].idf_
        return Xt

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> AnomalyDetector:
        """Load a saved detector. joblib uses pickle, which can run code while loading:
        ONLY load model files this project created itself."""
        return joblib.load(path)
