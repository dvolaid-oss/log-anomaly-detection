"""Isolation Forest (Liu et al., 2008).

Idea: build many random trees, each splitting the data on a random event at a random value.
Anomalies are few and different, so they get isolated (end up alone in a leaf) after only a
few splits; normal blocks sit in dense crowds and need many splits.

Score = how quickly a block gets isolated, averaged over all trees (we flip scikit-learn's
sign so that higher = more anomalous). No distance metric, no assumptions about the data's
shape, and fast: each tree only looks at a small random sample (256 blocks by default).

Limitation: trees can only split on events that VARY in the training data. An event that
never occurred in training is invisible to the forest (the rule baseline and PCA do catch it).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from logad.models.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        transform: str = "event_count",
        scale: bool = False,
        seed: int = 42,
    ):
        super().__init__(transform=transform, scale=scale, seed=seed)
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def _fit(self, X: np.ndarray) -> None:
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.seed,
            n_jobs=-1,  # use all CPU cores
        )
        self.forest_.fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: lower = more abnormal → negate so higher = more anomalous
        return -self.forest_.score_samples(X)
