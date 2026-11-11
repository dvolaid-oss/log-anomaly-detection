"""Evaluation charts (PNG, for the README and reports).

Style rules (from the project's chart guidelines): one fixed colour per model in a fixed
order, thin 2px lines, a light grid, a legend plus direct labels so identity never depends
on colour alone, and a different line style per model as a second cue. The colours were
checked for colour-blind separation; aqua and yellow are light, so labels/tables back them up.
More models than colours would need a new design (never generated extra hues).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow: fixed order
LINESTYLES = ["-", "--", ":", "-."]
NORMAL, ANOMALY = SERIES[0], SERIES[1]

LABELS = {
    "baseline": "Rule baseline",
    "pca": "PCA",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
}
SHORT = {"baseline": "Baseline", "pca": "PCA", "isolation_forest": "IForest", "autoencoder": "AE"}
# Where each model's threshold label sits relative to its dot (points), chosen to stay
# clear of the curves: (dx, dy, horizontal alignment).
LABEL_OFFSETS = {
    "baseline": (8, -4, "left"),
    "pca": (-10, -16, "right"),
    "isolation_forest": (-10, -44, "right"),
    "autoencoder": (-10, -30, "right"),
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_pr_curves(y_test, scores: dict, results: dict, base_rate: float, path: Path) -> Path:
    """Precision-recall curves on the test split, with each model's chosen threshold marked."""
    fig, ax = _figure(figsize=(7.5, 5))
    _style(ax)
    for i, (model, s) in enumerate(scores.items()):
        precision, recall, _ = precision_recall_curve(y_test, s)
        r = results[model]
        name = LABELS.get(model, model)
        ax.plot(
            recall,
            precision,
            color=SERIES[i],
            linestyle=LINESTYLES[i],
            linewidth=2,
            label=f"{name} (PR-AUC {r['pr_auc']:.3f})",
        )
        ax.plot(
            r["recall"],
            r["precision"],
            marker="o",
            markersize=9,
            color=SERIES[i],
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
        dx, dy, ha = LABEL_OFFSETS.get(model, (-10, -16 - 14 * i, "right"))
        if dx > 0 and r["recall"] > 0.75:  # a label to the right would run off the chart
            dx, dy, ha = -10, 10, "right"
        ax.annotate(
            f"{SHORT.get(model, name)} F1 {r['f1']:.2f}",
            (r["recall"], r["precision"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=9,
            color=TEXT,
        )
    ax.axhline(base_rate, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    ax.annotate(
        f"random guessing ({base_rate:.3f})",
        (0.02, base_rate),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall: share of anomalies caught")
    ax.set_ylabel("Precision: share of alerts that are real")
    ax.set_title("Precision vs recall on the test split (dots = chosen thresholds)", loc="left")
    # Legend BELOW the chart: curves differ per dataset, so no spot inside is always empty.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
    )
    return _save(fig, path)


def plot_score_distributions(y_test, scores: dict, results: dict, path: Path) -> Path:
    """Per model: score histograms of normal vs anomalous test blocks + the threshold."""
    y = np.asarray(y_test, dtype=bool)
    fig, axes = _figure(1, len(scores), figsize=(4.0 * len(scores), 3.8))
    for ax, (model, s) in zip(np.atleast_1d(axes), scores.items(), strict=True):
        _style(ax)
        s = np.asarray(s)
        bins = np.linspace(s.min(), np.percentile(s, 99.9), 50)
        for mask, color, name in [(~y, NORMAL, "normal"), (y, ANOMALY, "anomaly")]:
            ax.hist(
                np.clip(s[mask], bins[0], bins[-1]),
                bins=bins,
                # each class sums to 1, so the two colours are comparable despite 1.5% anomalies
                weights=np.full(int(mask.sum()), 1 / max(int(mask.sum()), 1)),
                histtype="step",
                linewidth=2,
                color=color,
                label=name,
            )
        threshold = results[model]["threshold"]
        ax.axvline(threshold, color=TEXT, linestyle=(0, (4, 4)), linewidth=1)
        ax.annotate(
            "threshold",
            (threshold, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
        )
        ax.set_yscale("log")
        ax.set_title(LABELS.get(model, model), loc="left", fontsize=11)
        ax.set_xlabel("anomaly score")
    np.atleast_1d(axes)[0].set_ylabel("share of the class's blocks (log)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Score distributions on the test split: normal vs anomalous blocks",
        x=0.01,
        ha="left",
        color=TEXT,
    )
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
