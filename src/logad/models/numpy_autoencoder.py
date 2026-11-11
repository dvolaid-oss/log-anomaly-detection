"""The trained autoencoder, re-implemented in plain NumPy (no PyTorch needed to RUN it).

A trained network is just a few weight matrices. Scoring a block means multiplying by them
and applying ReLU in between, which NumPy does just as well. Copying the weights out of the
PyTorch model lets the online dashboard use the autoencoder without installing PyTorch
(a ~124 MB download). Scores match the PyTorch model to float precision (tested).
"""

from __future__ import annotations

import numpy as np

from logad.models.base import AnomalyDetector


class NumpyAutoencoder(AnomalyDetector):
    name = "autoencoder"

    def __init__(self, transformer, layers: list[tuple[np.ndarray, np.ndarray]], seed: int = 42):
        # No super().__init__: the transformer arrives already FITTED with the PyTorch model.
        self.transform, self.scale, self.seed = "copied", False, seed
        self.transformer = transformer
        self.layers = layers  # [(weights, bias), ...]: ReLU after every layer but the last

    @classmethod
    def from_torch(cls, detector) -> NumpyAutoencoder:
        """Copy the weights out of a fitted PyTorch AutoencoderDetector."""
        linear = [m for m in detector.model_ if hasattr(m, "weight")]
        layers = [
            (m.weight.detach().numpy().T.astype(float), m.bias.detach().numpy().astype(float))
            for m in linear
        ]
        return cls(detector.transformer, layers, detector.seed)

    def _rebuild(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=float)
        for i, (weights, bias) in enumerate(self.layers):
            h = h @ weights + bias
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)  # ReLU
        return h

    def _score(self, X: np.ndarray) -> np.ndarray:
        return ((self._rebuild(X) - X) ** 2).mean(axis=1)  # same MSE as the PyTorch model

    def _fit(self, X: np.ndarray) -> None:
        raise NotImplementedError("Train the PyTorch AutoencoderDetector, then use from_torch().")
