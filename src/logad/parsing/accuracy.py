"""Parsing accuracy against hand-labelled templates (LogHub ground truth).

Grouping Accuracy (GA) is the standard metric from the LogPAI parser benchmark
(Zhu et al., "Tools and Benchmarks for Automated Log Parsing", 2019):

    A message counts as correctly parsed only if the group the parser put it in contains
    EXACTLY the same messages as its true group: no missing members, no extra ones.
    GA = correctly parsed messages / all messages.

The template *names* don't matter, only how messages are grouped.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def grouping_accuracy(true_ids: Sequence, predicted_ids: Sequence) -> float:
    """Share of messages whose predicted group equals their true group exactly."""
    if len(true_ids) != len(predicted_ids):
        raise ValueError("true_ids and predicted_ids must have the same length")
    if len(true_ids) == 0:
        raise ValueError("Cannot compute accuracy on zero messages")

    df = pd.DataFrame({"true": list(true_ids), "pred": list(predicted_ids)})
    true_sizes = df["true"].value_counts()

    correct = 0
    for _, group in df.groupby("pred"):
        true_in_group = group["true"].unique()
        # Correct only if the predicted group is "pure" (one true ID) AND complete
        # (it holds every message with that true ID).
        if len(true_in_group) == 1 and true_sizes[true_in_group[0]] == len(group):
            correct += len(group)
    return correct / len(df)
