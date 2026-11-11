import numpy as np
import pytest

pytest.importorskip("torch")  # PyTorch is optional: skip these tests if it isn't installed

from logad.models import build_models  # noqa: E402
from logad.models.autoencoder import AutoencoderDetector  # noqa: E402
from logad.models.base import AnomalyDetector  # noqa: E402


def normal_blocks(n=400, seed=0):
    """Events 0-2 always together in fixed proportions, event 3 never happens."""
    k = np.random.default_rng(seed).integers(1, 4, size=n)
    return np.column_stack([3 * k, 3 * k, 3 * k, np.zeros(n)]).astype(float)


NORMAL = np.array([[6.0, 6.0, 6.0, 0.0]])
ODD_MIX = np.array([[9.0, 0.0, 1.0, 0.0]])
RARE_EVENT = np.array([[6.0, 6.0, 6.0, 1.0]])


@pytest.fixture(scope="module")
def fitted():
    return AutoencoderDetector(seed=0, epochs=40).fit(normal_blocks())


def test_scores_unusual_blocks_higher(fitted):
    normal = fitted.score(NORMAL)[0]
    assert fitted.score(ODD_MIX)[0] > normal
    # unlike Isolation Forest, it also notices an event never seen in training
    assert fitted.score(RARE_EVENT)[0] > normal


def test_training_loss_goes_down(fitted):
    train_losses = [train for train, _ in fitted.history_]
    assert train_losses[-1] < train_losses[0]


def test_early_stopping_ends_before_the_epoch_limit():
    # Constant data is learned almost at once, so the held-out loss stops improving.
    X = np.tile([[3.0, 3.0, 3.0, 0.0]], (200, 1))
    detector = AutoencoderDetector(seed=0, epochs=200, patience=2).fit(X)
    assert detector.epochs_trained < 200


def test_same_seed_same_scores():
    X = normal_blocks(200)
    a = AutoencoderDetector(seed=5, epochs=5).fit(X).score(X)
    b = AutoencoderDetector(seed=5, epochs=5).fit(X).score(X)
    assert np.allclose(a, b)


def test_save_and_load(tmp_path, fitted):
    path = tmp_path / "autoencoder.joblib"
    fitted.save(path)
    loaded = AnomalyDetector.load(path)
    assert np.allclose(loaded.score(ODD_MIX), fitted.score(ODD_MIX))


def test_built_from_config():
    config = {"models": {"autoencoder": {"hidden": 16, "bottleneck": 4, "scale": True}}}
    detector = build_models(config)["autoencoder"]
    assert (detector.hidden, detector.bottleneck, detector.scale) == (16, 4, True)
