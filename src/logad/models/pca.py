"""PCA detector (Xu et al., "Detecting Large-Scale System Problems by Mining Console Logs",
SOSP 2009): the classic method for HDFS logs.

Idea: normal blocks follow a few common patterns (e.g. "receive → respond → store" in fixed
proportions). PCA finds those patterns as the directions that explain most of the variance
in the training data (here: enough components to keep 95% of it).

Score = squared prediction error (SPE): project a block onto the "normal" directions, map it
back, and measure how much was lost. A normal block is rebuilt almost perfectly (small
error); a block with an unusual mix of events is not (large error).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from logad.models.base import AnomalyDetector


class PCADetector(AnomalyDetector):
    name = "pca"

    def __init__(
        self,
        variance_kept: float = 0.95,
        transform: str = "tfidf",
        scale: bool = False,
        seed: int = 42,
    ):
        super().__init__(transform=transform, scale=scale, seed=seed)
        self.variance_kept = variance_kept

    def _fit(self, X: np.ndarray) -> None:
        # PCA centres the data itself (subtracts the training mean).
        self.pca_ = PCA(n_components=self.variance_kept, svd_solver="full", random_state=self.seed)
        self.pca_.fit(X)

    def _rebuild(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.inverse_transform(self.pca_.transform(X))

    def _score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self._rebuild(X)) ** 2).sum(axis=1)

    @property
    def n_components(self) -> int:
        return int(self.pca_.n_components_)
