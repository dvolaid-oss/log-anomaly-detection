"""Explanations: why was a block flagged?

Two kinds of reasons, both readable without knowing any maths:

1. **Rare events** (from the training data): the block contains an event that appeared in
   very few training blocks, e.g. "Contains E33 'Unexpected error trying to delete block …',
   seen in 0.04% of training blocks".

2. **Unusual mix** (from a model that REBUILDS blocks: PCA or the autoencoder): the model
   rebuilds each block from the patterns of normal blocks. Where the rebuilt block differs
   most from the real one, we report the event and both counts: "Missing: normal blocks like
   this have ~3x 'PacketResponder … terminating', this one has 0" or "Extra: 5x 'Got
   exception while serving …', expected ~0".
   This is how PCA catches blocks that are suspicious because something is *absent*.

3. **High counts** (from the training data): an event happens far more often than in 99% of
   training groups, e.g. "High count: 50x 'Failed password for …'; 99% of training windows
   have at most 3". This is the natural way to describe bursts like brute force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RARE_SHARE = 0.001  # events in fewer than 0.1% of training blocks count as "rare"


@dataclass
class Reason:
    kind: str  # "rare", "missing", "extra" or "unknown"
    event: str
    text: str
    weight: float  # how much this reason matters (for sorting; higher = more important)


def short(template: str, limit: int = 60) -> str:
    return template if len(template) <= limit else template[: limit - 3] + "..."


class Explainer:
    """Builds reasons for one block from a fitted detector and training statistics.

    "Unusual mix" reasons need a detector that rebuilds blocks (PCA, autoencoder); for other
    detectors only the rare-event reasons are given.
    """

    def __init__(
        self, detector, events: list[str], templates: dict[str, str], train_share, train_p99=None
    ):
        self.detector = detector
        self.events = list(events)
        self.templates = templates
        self.train_share = np.asarray(train_share, dtype=float)
        # 99th percentile of each event's count per training group (None: no "high" reasons)
        self.train_p99 = None if train_p99 is None else np.asarray(train_p99, dtype=float)

    def _name(self, i: int) -> str:
        event = self.events[i]
        return f"{event} '{short(self.templates.get(event, '?'))}'"

    def rare_reasons(self, counts: np.ndarray) -> list[Reason]:
        reasons = []
        for i in np.flatnonzero(counts > 0):
            share = self.train_share[i]
            if share < RARE_SHARE:
                seen = "never seen" if share == 0 else f"seen in {100 * share:.2g}%"
                reasons.append(
                    Reason(
                        "rare",
                        self.events[i],
                        f"Contains {self._name(i)}: {seen} of training blocks.",
                        weight=-np.log10(max(share, 1e-9)),
                    )
                )
        return sorted(reasons, key=lambda r: -r.weight)

    def mix_reasons(self, counts: np.ndarray, top: int = 3) -> list[Reason]:
        """Events where the block differs most from what the model expected, as counts."""
        if not getattr(self.detector, "rebuilds", False):
            return []
        _, expected = self.detector.reconstruction(counts.reshape(1, -1))
        expected = expected[0]
        # Rank by the difference in COUNTS, weighted by TF-IDF rarity when the model uses it.
        # (Not by the model's own scaled residual: scaling stretches rare events, so a 0.01
        # count difference could outrank a 300-line burst. For PCA, which has no scaling,
        # this is exactly its own residual.)
        tfidf = self.detector.transformer.named_steps.get("tfidf")
        weight = tfidf.idf_ if tfidf is not None else np.ones(len(counts))
        importance = ((counts - expected) * weight) ** 2
        total = float(importance.sum()) or 1.0

        reasons = []
        for i in np.argsort(-importance)[:top]:
            share = float(importance[i]) / total
            difference = abs(counts[i] - expected[i])
            if share < 0.05 or difference < max(0.75, 0.25 * expected[i]):  # too small to matter
                continue
            found, exp = int(counts[i]), float(expected[i])
            if found < exp:
                kind = "missing"
                text = f"Missing: expected ~{exp:.1f}x {self._name(i)}, found {found}."
            else:
                kind = "extra"
                text = f"Extra: found {found}x {self._name(i)}, expected ~{exp:.1f}."
            reasons.append(Reason(kind, self.events[i], text, weight=share))
        return reasons

    def high_reasons(self, counts: np.ndarray) -> list[Reason]:
        """Events counted far more often than in 99% of training groups (e.g. bursts)."""
        if self.train_p99 is None:
            return []
        reasons = []
        for i in np.flatnonzero(counts > 0):
            p99 = self.train_p99[i]
            if counts[i] > p99 and counts[i] - p99 >= 3:
                reasons.append(
                    Reason(
                        "high",
                        self.events[i],
                        f"High count: {int(counts[i])}x {self._name(i)}; 99% of training "
                        f"groups have at most {p99:g}.",
                        weight=float(counts[i] / max(p99, 1.0)),
                    )
                )
        return sorted(reasons, key=lambda r: -r.weight)

    def explain(self, counts, unknown_lines: int = 0) -> list[Reason]:
        """All reasons for one block, most important first."""
        counts = np.asarray(counts, dtype=float)
        reasons = self.rare_reasons(counts)
        for extra in (self.high_reasons(counts), self.mix_reasons(counts)):
            already = {r.event for r in reasons}  # say each event's story only once
            reasons += [r for r in extra if r.event not in already]
        if unknown_lines:
            reasons.append(
                Reason(
                    "unknown",
                    "?",
                    f"{unknown_lines} line(s) match no known template (new kind of message).",
                    weight=0.0,
                )
            )
        return reasons
