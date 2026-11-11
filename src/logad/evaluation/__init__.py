"""Evaluation: how well would each detector work in production?

`evaluate_scores(config)` reads `scores.parquet` (written by `logad train`) and, per model:

1. picks the alert threshold with the best F1 on the VALIDATION split, the one place where
   labels influence the system (one number per model);
2. applies that threshold ONCE to the untouched TEST split: precision, recall, F1, alert
   counts, ROC-AUC, PR-AUC, and bootstrap 95% confidence intervals;
3. saves `thresholds.json` (for the dashboard), charts and `reports/<dataset>/results.md`.

Rule: the test split is for the final grade only. Nothing is tuned after looking at it.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from logad.evaluation.metrics import evaluate_model
from logad.evaluation.plots import LABELS, plot_pr_curves, plot_score_distributions

SCORES_FILE = "scores.parquet"
THRESHOLDS_FILE = "thresholds.json"


def evaluate_scores(config: dict[str, Any], n_resamples: int = 1000) -> pd.DataFrame:
    """Threshold on validation, measure on test, write reports. Returns one row per model."""
    artifacts = Path(config["paths"]["artifacts_dir"])
    scores_path = artifacts / SCORES_FILE
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found. Run `logad train` first.")
    df = pd.read_parquet(scores_path)
    if df["label"].isna().all():
        raise ValueError(
            "This dataset has no labels, so it can't be evaluated. "
            "Use a labelled config such as configs/hdfs.yaml."
        )

    models = [c.removeprefix("score_") for c in df.columns if c.startswith("score_")]
    val = df[(df["split"] == "validation") & df["label"].notna()]
    test = df[(df["split"] == "test") & df["label"].notna()]
    y_val, y_test = val["label"].astype(int), test["label"].astype(int)

    seed = config.get("seed", 42)
    results = {
        m: evaluate_model(
            m, y_val, val[f"score_{m}"], y_test, test[f"score_{m}"], n_resamples, seed
        ).as_dict()
        for m in models
    }

    thresholds = {
        m: {"threshold": r["threshold"], "validation_f1": r["validation_f1"]}
        for m, r in results.items()
    }
    (artifacts / THRESHOLDS_FILE).write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    reports = Path(config["paths"]["reports_dir"])
    test_scores = {m: test[f"score_{m}"].to_numpy() for m in models}
    plot_pr_curves(y_test, test_scores, results, float(y_test.mean()), reports / "pr_curves.png")
    plot_score_distributions(y_test, test_scores, results, reports / "score_distributions.png")
    breakdown = attack_breakdown(config, test, results)
    write_results_report(results, y_test, config, breakdown)
    table = pd.DataFrame(results.values())
    table.attrs["breakdown"] = breakdown  # per-attack table (synthetic auth.log only)
    return table


def attack_breakdown(config: dict[str, Any], test: pd.DataFrame, results: dict):
    """Per attack type: how many TEST groups contain it, and how many each model flags.

    Only for datasets whose labels file names the attack type per line (the synthetic
    auth.log). Returns None otherwise. A group can contain several attack types.
    """
    data = config.get("data", {})
    label_file = data.get("label_file")
    if not label_file:
        return None
    labels_path = Path(config["paths"]["raw_dir"]) / label_file
    if not labels_path.is_file():
        return None
    attacks = pd.read_csv(labels_path, keep_default_na=False)
    if "attack" not in attacks.columns:
        return None

    from logad.features import group_expression  # local import: avoids a cycle

    events = Path(config["paths"]["artifacts_dir"]) / "events.parquet"
    con = duckdb.connect()
    con.register("attacks", attacks[attacks["attack"] != ""])
    con.register("test_groups", test[["group_id"]])
    pairs = con.sql(f"""
        SELECT DISTINCT {group_expression(config)} AS group_id, a.attack
        FROM '{events.as_posix()}' e JOIN attacks a USING (line_id)
    """).df()
    pairs = pairs[pairs["group_id"].isin(test["group_id"])]

    alerts = {
        m: set(test.loc[test[f"score_{m}"] >= r["threshold"], "group_id"])
        for m, r in results.items()
    }
    rows = []
    for attack, groups in pairs.groupby("attack")["group_id"]:
        groups = set(groups)
        row = {"attack": attack, "groups": len(groups)}
        row.update({m: len(groups & flagged) for m, flagged in alerts.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("groups", ascending=False, ignore_index=True)


def _ci(interval) -> str:
    return f"{interval[0]:.3f}–{interval[1]:.3f}"


def write_results_report(
    results: dict, y_test: pd.Series, config: dict[str, Any], breakdown=None
) -> Path:
    reports = Path(config["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    n_test, n_anomalies = len(y_test), int(y_test.sum())
    best = max(results.values(), key=lambda r: r["f1"])
    lines = [
        f"# Results: {config['dataset']}",
        "",
        f"Generated by `logad evaluate` on {date.today():%Y-%m-%d}.",
        "",
        "- Models were trained on the **train** split (oldest 60% of blocks) **without labels**.",
        "- Each model's alert threshold is the one with the best F1 on the **validation** split",
        "  (next 20%). This is the only place labels influence the system.",
        f"- Results below are on the **test** split (newest 20%: {n_test:,} blocks, "
        f"{n_anomalies:,} anomalous = {100 * n_anomalies / n_test:.2f}%), evaluated once.",
        "- Brackets: 95% bootstrap confidence intervals (1,000 resamples of the test set).",
        "",
        "| Model | Precision | Recall | F1 | Caught | False alarms | Missed | ROC-AUC | PR-AUC |",
        "|-------|----------:|-------:|---:|-------:|-------------:|-------:|--------:|-------:|",
    ]
    for m, r in results.items():
        lines.append(
            f"| {LABELS.get(m, m)} "
            f"| {r['precision']:.3f} [{_ci(r['precision_ci'])}] "
            f"| {r['recall']:.3f} [{_ci(r['recall_ci'])}] "
            f"| **{r['f1']:.3f}** [{_ci(r['f1_ci'])}] "
            f"| {r['tp']:,} | {r['fp']:,} | {r['fn']:,} "
            f"| {r['roc_auc']:.3f} | {r['pr_auc']:.3f} |"
        )
    lines += [
        "",
        f"Best F1 on test: **{LABELS.get(best['model'], best['model'])}** ({best['f1']:.3f}). "
        "Where two models' F1 intervals overlap, the test set can't tell them apart reliably.",
        "",
        "Thresholds (chosen on validation, in `artifacts/.../thresholds.json`):",
        "",
    ]
    lines += [
        f"- {LABELS.get(m, m)}: {r['threshold']:.4g} (validation F1 {r['validation_f1']:.3f})"
        for m, r in results.items()
    ]
    if breakdown is not None and not breakdown.empty:
        models = [m for m in results if m in breakdown.columns]
        lines += [
            "",
            "## Caught per attack type (test split)",
            "",
            "Groups (IP × window) containing each attack type, and how many each model flagged.",
            "",
            "| Attack | Groups | " + " | ".join(LABELS.get(m, m) for m in models) + " |",
            "|--------|-------:|" + "|".join("-----:" for _ in models) + "|",
        ]
        for row in breakdown.itertuples(index=False):
            row = row._asdict()
            cells = " | ".join(f"{row[m]} ({100 * row[m] / row['groups']:.0f}%)" for m in models)
            lines.append(f"| {row['attack']} | {row['groups']} | {cells} |")
    lines += [
        "",
        "![Precision-recall curves](pr_curves.png)",
        "",
        "![Score distributions](score_distributions.png)",
        "",
    ]
    path = reports / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
