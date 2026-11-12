"""README figures comparing normal and anomalous data across datasets (`logad charts`).

Four figures, each with one panel per dataset, all on the TEST split (never trained on),
using each dataset's best model on VALIDATION:

1. PCA scatter plot   : groups projected to 2D (PCA fitted on the train split), by true label
2. Box plot           : anomaly scores, normal vs anomaly (+ SSH: one box per attack type)
3. Distribution plot  : KDE density curves of the scores, with the alert threshold
4. Bubble chart       : event types; x/y = share of normal/anomalous groups containing them,
                        size = number of lines

Plus `facts.json`: the numbers the README explanations quote, so they come from the data.
Style follows the project's chart rules: fixed validated colours (blue normal, orange
anomaly), thin marks, recessive grid, direct labels, no dual axes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from logad.evaluation.plots import GRID, LABELS, MUTED, SURFACE, TEXT, _figure, _save, _style
from logad.features import event_columns, group_expression

NORMAL, ANOMALY, NEUTRAL = "#2a78d6", "#eb6834", "#9a9890"
NAMES = {"hdfs": "HDFS (Hadoop blocks)", "bgl": "BGL (10-min windows)", "auth": "SSH (IP x 10-min)"}


@dataclass
class View:
    """Everything the charts need for one dataset."""

    dataset: str
    X_train: np.ndarray
    test: pd.DataFrame  # group_id, label, score, n_lines + event columns
    events: list[str]
    model: str
    threshold: float
    templates: dict[str, str]
    attacks: pd.Series | None  # group_id -> "brute_force;..." (SSH only)
    config: dict[str, Any]


def load_view(config: dict[str, Any]) -> View:
    artifacts = Path(config["paths"]["artifacts_dir"])
    features = pd.read_parquet(artifacts / "features.parquet")
    events = event_columns(features.columns)
    scores = pd.read_parquet(artifacts / "scores.parquet")
    thresholds = json.loads((artifacts / "thresholds.json").read_text(encoding="utf-8"))
    model = max(thresholds, key=lambda m: thresholds[m]["validation_f1"])  # best on validation
    templates = pd.read_csv(artifacts / "templates.csv", keep_default_na=False)

    test = features[(features["split"] == "test") & features["label"].notna()].copy()
    test["label"] = test["label"].astype(int)
    test["n_lines"] = test[events].sum(axis=1)
    test = test.merge(scores[["group_id", f"score_{model}"]], on="group_id")
    test = test.rename(columns={f"score_{model}": "score"})
    return View(
        dataset=config["dataset"],
        X_train=features.loc[features["split"] == "train", events].to_numpy(dtype=float),
        test=test.reset_index(drop=True),
        events=events,
        model=model,
        threshold=float(thresholds[model]["threshold"]),
        templates=dict(zip(templates["event_id"], templates["template"], strict=True)),
        attacks=_group_attacks(config),
        config=config,
    )


def _group_attacks(config: dict[str, Any]) -> pd.Series | None:
    """For datasets whose labels file names the attack type per line (synthetic SSH)."""
    label_file = config.get("data", {}).get("label_file")
    path = Path(config["paths"]["raw_dir"]) / label_file if label_file else None
    if path is None or not path.is_file():
        return None
    labels = pd.read_csv(path, keep_default_na=False)
    if "attack" not in labels.columns:
        return None
    con = duckdb.connect()
    con.register("attacks", labels[labels["attack"] != ""])
    events = Path(config["paths"]["artifacts_dir"]) / "events.parquet"
    pairs = con.sql(f"""
        SELECT DISTINCT {group_expression(config)} AS group_id, a.attack
        FROM '{events.as_posix()}' e JOIN attacks a USING (line_id)
    """).df()
    return pairs.groupby("group_id")["attack"].agg(lambda a: ";".join(sorted(a)))


def _log_scores(scores: pd.Series) -> np.ndarray:
    positive = scores[scores > 0]
    floor = float(positive.min()) if len(positive) else 1e-9
    return np.log10(np.clip(scores.to_numpy(dtype=float), floor, None))


def _panel_title(ax, view: View, extra: str = "") -> None:
    ax.set_title(f"{NAMES.get(view.dataset, view.dataset)}{extra}", loc="left", fontsize=11)


def _threshold_line(ax, threshold: float, at: float = 0.0, vertical=False, ha="center") -> None:
    """Dashed line at the alert threshold (scores are drawn on a log10 scale)."""
    position = np.log10(threshold)
    style = {"color": TEXT, "linestyle": (0, (4, 4)), "linewidth": 1}
    label = {"fontsize": 9, "color": TEXT, "textcoords": "offset points"}
    if vertical:
        ax.axvline(position, **style)
        ax.annotate(
            "threshold",
            (position, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            **label,
        )
    else:
        ax.axhline(position, **style)
        ax.annotate("threshold", (at, position), xytext=(0, 4), ha=ha, **label)


def _legend(fig, items: list[tuple[str, str]]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=8, label=t) for t, c in items
    ]
    fig.legend(handles=handles, loc="upper right", ncol=len(items), frameon=False, fontsize=9)


# --------------------------------------------------------------------------- 1. PCA scatter
def pca_scatter(views: list[View], path: Path, max_normal: int = 4000, seed: int = 42) -> dict:
    """Test groups in 2D: PCA on log(1 + counts), standardised, fitted on the TRAIN split."""
    rng = np.random.default_rng(seed)
    fig, axes = _figure(1, len(views), figsize=(5.2 * len(views), 4.6))
    facts = {}
    for ax, view in zip(np.atleast_1d(axes), views, strict=True):
        _style(ax)
        train = view.X_train
        if len(train) > 60_000:  # a sample is plenty to find the 2 main directions
            train = train[rng.choice(len(train), 60_000, replace=False)]
        scaler = StandardScaler().fit(np.log1p(train))
        pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(np.log1p(train)))
        xy = pca.transform(scaler.transform(np.log1p(view.test[view.events].to_numpy(float))))
        # Many groups have identical counts and would sit on exactly the same spot: a tiny
        # random offset (jitter) makes stacked points visible. Positions stay honest overall.
        spread = np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0)
        xy = xy + rng.normal(0, 0.02 * np.maximum(spread, 1e-6), size=xy.shape)
        is_anomaly = view.test["label"].to_numpy() == 1
        normal_idx = np.flatnonzero(~is_anomaly)
        if len(normal_idx) > max_normal:
            normal_idx = rng.choice(normal_idx, max_normal, replace=False)
        ax.scatter(*xy[normal_idx].T, s=7, color=NORMAL, alpha=0.35, linewidths=0)
        ax.scatter(*xy[is_anomaly].T, s=14, color=ANOMALY, alpha=0.75, linewidths=0)
        # Symmetric log scale: linear near 0, logarithmic far out, so the dense centre and
        # the extreme groups are both visible.
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        var = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"principal component 1 ({var[0]:.0f}% of variance)")
        ax.set_ylabel(f"principal component 2 ({var[1]:.0f}% of variance)")
        _panel_title(ax, view)
        facts[view.dataset] = {
            "normal_shown": int(len(normal_idx)),
            "normal_total": int((~is_anomaly).sum()),
            "anomalies_shown": int(is_anomaly.sum()),
            "variance_pct": [round(float(v), 1) for v in var],
        }
    _legend(fig, [("normal", NORMAL), ("anomaly", ANOMALY)])
    fig.suptitle(
        "PCA scatter plot of test groups (true labels; symmetric log axes)",
        x=0.01,
        ha="left",
        color=TEXT,
    )
    _save(fig, path)
    return facts


# --------------------------------------------------------------------------- 2. box plots
def _boxes(ax, groups: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    parts = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": TEXT, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": MUTED,
            "alpha": 0.4,
            "markeredgewidth": 0,
        },
    )
    for box, color in zip(parts["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.75)
        box.set_edgecolor(SURFACE)


def box_plots(views: list[View], path: Path) -> dict:
    """Anomaly scores (log scale) of the best model, normal vs anomaly, + threshold."""
    fig, axes = _figure(1, len(views), figsize=(4.6 * len(views), 4.4))
    facts = {}
    for ax, view in zip(np.atleast_1d(axes), views, strict=True):
        _style(ax)
        logs = _log_scores(view.test["score"])
        is_anomaly = view.test["label"].to_numpy() == 1
        _boxes(ax, [logs[~is_anomaly], logs[is_anomaly]], ["normal", "anomaly"], [NORMAL, ANOMALY])
        _threshold_line(ax, view.threshold, at=1.5)
        ax.set_ylabel(f"anomaly score, log10 ({LABELS.get(view.model, view.model)})")
        _panel_title(ax, view)
        normal_scores = view.test.loc[~is_anomaly, "score"]
        anomaly_scores = view.test.loc[is_anomaly, "score"]
        facts[view.dataset] = {
            "model": view.model,
            "median_normal": float(normal_scores.median()),
            "median_anomaly": float(anomaly_scores.median()),
            "threshold": view.threshold,
            "share_normal_above": float((normal_scores >= view.threshold).mean()),
            "share_anomaly_above": float((anomaly_scores >= view.threshold).mean()),
        }
    fig.suptitle(
        "Box plot of anomaly scores, normal vs anomaly (test split)", x=0.01, ha="left", color=TEXT
    )
    _save(fig, path)
    return facts


def attack_box_plot(view: View, path: Path) -> dict:
    """SSH only: one box per attack type, next to normal windows."""
    test = view.test.assign(attack=view.test["group_id"].map(view.attacks).fillna(""))
    test = test.assign(attack=test["attack"].str.split(";")).explode("attack")
    test.loc[test["label"] == 0, "attack"] = "normal"
    test = test[test["attack"] != ""]
    logs = pd.Series(_log_scores(test["score"]), index=test.index)
    order = ["normal"] + sorted(
        set(test["attack"]) - {"normal"}, key=lambda a: -logs[test["attack"] == a].median()
    )
    fig, ax = _figure(figsize=(9, 4.6))
    _style(ax)
    groups = [logs[test["attack"] == a].to_numpy() for a in order]
    labels = [f"{a.replace('_', ' ')}\n(n={len(g)})" for a, g in zip(order, groups, strict=True)]
    _boxes(ax, groups, labels, [NORMAL] + [ANOMALY] * (len(order) - 1))
    _threshold_line(ax, view.threshold, at=len(order) + 0.45, ha="right")
    ax.set_ylabel(f"anomaly score, log10 ({LABELS.get(view.model, view.model)})")
    ax.set_title("Box plot of SSH anomaly scores per attack type (test split)", loc="left")
    _save(fig, path)
    return {
        a: {
            "windows": int((test["attack"] == a).sum()),
            "median": float(test.loc[test["attack"] == a, "score"].median()),
            "share_above": float((test.loc[test["attack"] == a, "score"] >= view.threshold).mean()),
        }
        for a in order
    }


# --------------------------------------------------------------------------- 3. KDE
def score_distributions(views: list[View], path: Path) -> dict:
    """Smooth (KDE) density curves of log scores, normal vs anomaly, + threshold."""
    fig, axes = _figure(1, len(views), figsize=(5.2 * len(views), 4.2))
    facts = {}
    for ax, view in zip(np.atleast_1d(axes), views, strict=True):
        _style(ax)
        logs = _log_scores(view.test["score"])
        is_anomaly = view.test["label"].to_numpy() == 1
        grid = np.linspace(logs.min(), logs.max(), 400)
        for mask, color, name in [
            (~is_anomaly, NORMAL, "normal"),
            (is_anomaly, ANOMALY, "anomaly"),
        ]:
            values = logs[mask]
            if len(np.unique(values)) < 2:
                continue
            density = gaussian_kde(values, bw_method=0.25)(grid)
            ax.fill_between(grid, density, color=color, alpha=0.18, linewidth=0)
            ax.plot(grid, density, color=color, linewidth=2, label=name)
        _threshold_line(ax, view.threshold, vertical=True)
        ax.set_xlabel(f"anomaly score, log10 ({LABELS.get(view.model, view.model)})")
        ax.set_ylabel("density")
        ax.set_yticks([])
        _panel_title(ax, view)
        facts[view.dataset] = {
            "missed": int((view.test.loc[is_anomaly, "score"] < view.threshold).sum()),
            "false_alarms": int((view.test.loc[~is_anomaly, "score"] >= view.threshold).sum()),
            "anomalies": int(is_anomaly.sum()),
            "normals": int((~is_anomaly).sum()),
        }
    _legend(fig, [("normal", NORMAL), ("anomaly", ANOMALY)])
    fig.suptitle(
        "Distribution plot (KDE) of anomaly scores (test split)", x=0.01, ha="left", color=TEXT
    )
    _save(fig, path)
    return facts


# --------------------------------------------------------------------------- 4. bubble chart
def bubble_charts(views: list[View], path: Path, floor: float = 0.01) -> dict:
    """Each ball = one event type. x/y = % of normal / anomalous test groups containing it."""
    fig, axes = _figure(1, len(views), figsize=(5.2 * len(views), 4.8))
    facts = {}
    for ax, view in zip(np.atleast_1d(axes), views, strict=True):
        _style(ax)
        present = view.test[view.events] > 0
        is_anomaly = view.test["label"].to_numpy() == 1
        pct_normal = 100 * present[~is_anomaly].mean()
        pct_anomaly = 100 * present[is_anomaly].mean()
        lines = view.test[view.events].sum()
        keep = (pct_normal > 0) | (pct_anomaly > 0)
        x = np.maximum(pct_normal[keep], floor)
        y = np.maximum(pct_anomaly[keep], floor)
        ratio = (y / x).rename("ratio")
        colors = np.where(ratio > 2, ANOMALY, np.where(ratio < 0.5, NORMAL, NEUTRAL))
        sizes = 15 + 900 * np.sqrt(lines[keep] / lines.max())
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors=SURFACE, linewidths=1.5)
        ax.plot([floor, 100], [floor, 100], color=GRID, linewidth=1.5, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(floor / 1.6, 160)
        ax.set_ylim(floor / 1.6, 160)
        ax.set_xlabel("% of NORMAL groups containing the event")
        ax.set_ylabel("% of ANOMALOUS groups containing it")
        # Label the strongest anomaly signals (common in anomalies, rare in normal groups).
        # Each label sits next to its ball; if it would collide with the label above it, it
        # moves down and a thin line connects it to its ball.
        top = ratio[(ratio > 2) & (y >= 1)].sort_values(ascending=False).head(3)
        placed: list[float] = []  # label heights so far, in log10 units
        for event in sorted(top.index, key=lambda e: -y[e]):
            height = np.log10(y[event])
            shift = 0
            while any(abs(height - shift * 0.15 - other) < 0.15 for other in placed):
                shift += 1
            placed.append(height - shift * 0.15)
            ax.annotate(
                event,
                (x[event], y[event]),
                xytext=(10, -14 * shift),
                textcoords="offset points",
                fontsize=9,
                color=TEXT,
                va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8} if shift else None,
            )
        ticks = [floor, 0.1, 1, 10, 100]
        labels = [f"≤{floor:g}", "0.1", "1", "10", "100"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.minorticks_off()
        _panel_title(ax, view)
        facts[view.dataset] = {
            "events": int(keep.sum()),
            "anomaly_signals": int((ratio > 2).sum()),
            "top": [
                {
                    "event": e,
                    "template": view.templates.get(e, "?"),
                    "pct_normal": round(float(pct_normal[e]), 3),
                    "pct_anomaly": round(float(pct_anomaly[e]), 2),
                }
                for e in top.index
            ],
        }
    _legend(
        fig, [("mostly in anomalies", ANOMALY), ("mostly in normal", NORMAL), ("both", NEUTRAL)]
    )
    fig.suptitle(
        "Bubble chart of event types: where does each message appear? (ball size = lines)",
        x=0.01,
        ha="left",
        color=TEXT,
    )
    _save(fig, path)
    return facts


# --------------------------------------------------------------------------- all + facts
def group_summary(view: View) -> dict:
    """Plain numbers comparing normal and anomalous test groups."""
    is_anomaly = view.test["label"] == 1
    event_types = (view.test[view.events] > 0).sum(axis=1)  # distinct event types per group
    return {
        "test_groups": int(len(view.test)),
        "anomalies": int(is_anomaly.sum()),
        "anomaly_pct": round(100 * float(is_anomaly.mean()), 2),
        "median_lines_normal": float(view.test.loc[~is_anomaly, "n_lines"].median()),
        "median_lines_anomaly": float(view.test.loc[is_anomaly, "n_lines"].median()),
        "median_event_types_normal": float(event_types[~is_anomaly].median()),
        "median_event_types_anomaly": float(event_types[is_anomaly].median()),
    }


def make_charts(configs: list[dict[str, Any]], out_dir: Path) -> dict:
    views = [load_view(c) for c in configs]
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = {
        "summary": {v.dataset: group_summary(v) for v in views},
        "pca_scatter": pca_scatter(views, out_dir / "pca_scatter.png"),
        "box_plot": box_plots(views, out_dir / "box_plot.png"),
        "distribution": score_distributions(views, out_dir / "distribution.png"),
        "bubble": bubble_charts(views, out_dir / "bubble_chart.png"),
    }
    ssh = [v for v in views if v.attacks is not None]
    if ssh:
        facts["attack_box_plot"] = attack_box_plot(ssh[0], out_dir / "box_plot_attacks.png")
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    return facts
