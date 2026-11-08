"""Transform raw event counts before a model sees them.

Both steps LEARN something from the data (IDF weights, column means/spreads), so they are
fitted on the TRAINING split only and then applied to every split. Fitting on all data
would leak information about the validation/test blocks into training.

Same `fit` / `transform` API as scikit-learn, so it plugs into any sklearn model.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURE_TYPES = {"event_count", "tfidf"}


def _to_float(X):
    # A named function (not a lambda), so a fitted pipeline can be saved with joblib.
    return np.asarray(X, dtype=float)


class TfidfWeighting(BaseEstimator, TransformerMixin):
    """Multiply each event's count by its inverse document frequency (IDF).

    IDF is high for events that appear in few blocks (rare → informative, like an error)
    and low for events in almost every block (common → little information):

        idf(event) = ln((1 + n_blocks) / (1 + blocks_containing_event)) + 1

    (the "smoothed" formula scikit-learn uses; the +1s avoid division by zero and keep
    every weight above zero).
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_blocks = X.shape[0]
        blocks_with_event = (X > 0).sum(axis=0)
        self.idf_ = np.log((1 + n_blocks) / (1 + blocks_with_event)) + 1
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * self.idf_


def make_transformer(feature_type: str = "event_count", scale: bool = False) -> Pipeline:
    """Build the (unfitted) transformer described by the config's `features` section."""
    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            f"features.type must be one of {sorted(FEATURE_TYPES)}, got {feature_type!r}"
        )
    steps = []
    if feature_type == "tfidf":
        steps.append(("tfidf", TfidfWeighting()))
    if scale:
        # Zero mean, unit variance per event. Columns that never vary keep scale 1.
        steps.append(("scale", StandardScaler()))
    if not steps:
        steps.append(("identity", FunctionTransformer(_to_float)))
    return Pipeline(steps)
