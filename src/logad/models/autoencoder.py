"""Autoencoder detector: a small neural network that learns to rebuild normal blocks.

    45 event features → 32 → 8 (bottleneck) → 32 → 45 rebuilt features

Trained on the train split only (no labels) to make its output equal its input. The narrow
bottleneck forces it to learn the typical structure of a block instead of copying it, so a
normal block is rebuilt well and an unusual one is not. Score = mean squared rebuilding
error per block.

Same idea as PCA (which is a *linear* autoencoder), but the ReLU layers let it learn
non-linear patterns too.

Early stopping: 10% of the TRAIN blocks are held out; training stops when their rebuilding
error stops improving, and the best weights are kept. The validation split stays untouched
for threshold selection.

PyTorch is optional (`pip install -r requirements-torch.txt`), so it is only imported when
this model is actually used.
"""

from __future__ import annotations

import numpy as np

from logad.models.base import AnomalyDetector


def _import_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The autoencoder needs PyTorch: pip install -r requirements-torch.txt"
        ) from error
    return torch


class AutoencoderDetector(AnomalyDetector):
    name = "autoencoder"

    def __init__(
        self,
        hidden: int = 32,
        bottleneck: int = 8,
        epochs: int = 30,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 3,
        holdout: float = 0.1,
        transform: str = "tfidf",
        scale: bool = True,  # neural networks train badly on unscaled counts
        seed: int = 42,
    ):
        super().__init__(transform=transform, scale=scale, seed=seed)
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.holdout = holdout

    def _build(self, n_features: int):
        nn = _import_torch().nn
        return nn.Sequential(
            nn.Linear(n_features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.bottleneck),
            nn.ReLU(),
            nn.Linear(self.bottleneck, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, n_features),
        )

    def _fit(self, X: np.ndarray) -> None:
        torch = _import_torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Hold out part of the TRAIN data for early stopping.
        order = rng.permutation(len(X))
        n_hold = max(1, int(len(X) * self.holdout))
        X_fit = torch.tensor(X[order[n_hold:]], dtype=torch.float32)
        X_hold = torch.tensor(X[order[:n_hold]], dtype=torch.float32)

        self.model_ = self._build(X.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        shuffler = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        self.history_: list[tuple[float, float]] = []  # (train loss, holdout loss) per epoch
        for _ in range(self.epochs):
            self.model_.train()
            total = 0.0
            for batch in torch.randperm(len(X_fit), generator=shuffler).split(self.batch_size):
                x = X_fit[batch]
                optimizer.zero_grad()
                loss = loss_fn(self.model_(x), x)  # target = the input itself
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)

            hold_loss = float(self._errors(X_hold).mean())
            self.history_.append((total / len(X_fit), hold_loss))
            if hold_loss < best_loss - 1e-6:
                best_loss, bad_epochs = hold_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break  # no improvement for `patience` epochs: stop
        self.model_.load_state_dict(best_state)
        self.model_.eval()

    def _errors(self, X):
        """Mean squared rebuilding error per row."""
        torch = _import_torch()
        X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            return ((self.model_(X) - X) ** 2).mean(dim=1).numpy()

    def _score(self, X: np.ndarray) -> np.ndarray:
        return self._errors(X).astype(float)

    @property
    def epochs_trained(self) -> int:
        return len(self.history_)
