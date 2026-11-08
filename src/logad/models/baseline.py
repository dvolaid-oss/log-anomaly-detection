"""Rule baseline: a block is as suspicious as the rarest event it contains.

During `fit` we measure, for each event, the share of training blocks that contain it.
A block's score is the rarity of its rarest event:

    rarity(event) = -log10(share of training blocks containing the event)

    event in every block      → share 1.0     → rarity 0
    event in 1 block in 1,000 → share 0.001   → rarity 3
    event never seen          → share 1/(n+1) → highest rarity

No labels, no tuning, and every score comes with a reason ("contains E33, seen in 0.03% of
training blocks"). The ML models must beat this to be worth their complexity.
"""

from __future__ import annotations

import numpy as np

from logad.models.base import AnomalyDetector


class RareEventDetector(AnomalyDetector):
    name = "baseline"

    def __init__(self, seed: int = 42, **_ignored):
        # Only presence matters here, so the rule always works on plain counts.
        super().__init__(transform="event_count", scale=False, seed=seed)

    def _fit(self, X: np.ndarray) -> None:
        share = (X > 0).mean(axis=0)
        floor = 1 / (len(X) + 1)  # events never seen in training get the highest rarity
        self.rarity_ = -np.log10(np.maximum(share, floor))

    def _score(self, X: np.ndarray) -> np.ndarray:
        # rarity of every event present in the block, 0 where absent → take the max
        return np.where(X > 0, self.rarity_, 0.0).max(axis=1)

    def rarest_event(self, X) -> np.ndarray:
        """Column index of the event that gave each block its score (for explanations)."""
        X = self.transformer.transform(X)
        return np.where(X > 0, self.rarity_, -1.0).argmax(axis=1)
