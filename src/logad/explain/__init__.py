"""Explanations: why was a block flagged?

Two kinds of reasons, both readable without knowing any maths:

1. **Rare events** (from the training data): the block contains an event that appeared in
   very few training blocks, e.g. "Contains E33 'Unexpected error trying to delete block …',
   seen in 0.04% of training blocks".

2. **Unusual mix** (from the PCA model): PCA rebuilds each block from the patterns of normal
   blocks. Where the rebuilt block differs most from the real one, we report the event and
   both counts: "Missing: normal blocks like this have ~3x 'PacketResponder … terminating',
   this one has 0" or "Extra: 5x 'Got exception while serving …', expected ~0".
   This is how PCA catches blocks that are suspicious because something is *absent*.
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
    """Builds reasons for one block from a fitted PCA detector and training statistics."""

    def __init__(self, pca_detector, events: list[str], templates: dict[str, str], train_share):
        self.detector = pca_detector
        self.events = list(events)
        self.templates = templates
        self.train_share = np.asarray(train_share, dtype=float)
        tfidf = pca_detector.transformer.named_steps.get("tfidf")
        # PCA works on TF-IDF values (count × idf); dividing by idf goes back to counts.
        self.idf = tfidf.idf_ if tfidf is not None else np.ones(len(self.events))

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
        """Events where the block differs most from what PCA expected, as counts."""
        x = self.detector.transformer.transform(counts.reshape(1, -1))
        rebuilt = self.detector.pca_.inverse_transform(self.detector.pca_.transform(x))
        residual = (x - rebuilt)[0]
        total = float((residual**2).sum()) or 1.0
        expected = np.maximum(rebuilt[0] / self.idf, 0)  # back to counts, no negatives

        reasons = []
        for i in np.argsort(-(residual**2))[:top]:
            share = float(residual[i] ** 2) / total
            if share < 0.05:  # ignore events that barely contribute
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

    def explain(self, counts, unknown_lines: int = 0) -> list[Reason]:
        """All reasons for one block, most important first."""
        counts = np.asarray(counts, dtype=float)
        rare = self.rare_reasons(counts)
        already = {r.event for r in rare}  # don't report a rare event a second time as "extra"
        reasons = rare + [r for r in self.mix_reasons(counts) if r.event not in already]
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
