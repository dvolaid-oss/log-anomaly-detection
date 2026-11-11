"""Feature engineering: `events.parquet` → one row of numbers per group.

A group is a session (HDFS: a block) or a fixed time window (BGL: e.g. one hour, set by
`grouping` in the config). For HDFS, every block becomes a vector of how many times each
event occurred:

                 E1  E2  E3 ... E45 | split  label
    blk_-1608...  3   1   3 ...   0 | train  0

This "event-count matrix" is the standard input for log anomaly detection models.

Blocks are also split by TIME (when each block first appears): the oldest share is
`train`, then `validation`, then `test`. Models learn from the past and are judged on the
future, like in production, so no information flows backwards in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from logad.features.transform import make_transformer

EVENTS_FILE = "events.parquet"
FEATURES_FILE = "features.parquet"
SPLITS = ("train", "validation", "test")
META_COLUMNS = ["group_id", "first_seen", "split", "label"]


@dataclass
class FeatureStats:
    blocks: int = 0
    events: int = 0
    split_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def event_columns(columns) -> list[str]:
    """Event columns (E1, E2, ...) in numeric order: E2 before E10."""
    events = [c for c in columns if c.startswith("E") and c[1:].isdigit()]
    return sorted(events, key=lambda c: int(c[1:]))


def assign_splits(n_rows: int, train: float, validation: float) -> np.ndarray:
    """Split labels for rows already sorted by time: first `train` share, then `validation`,
    the rest `test`."""
    if not (0 < train < 1 and 0 <= validation < 1 and train + validation < 1):
        raise ValueError("Need 0 < train < 1, 0 <= validation, and train + validation < 1")
    n_train = int(round(n_rows * train))
    n_validation = int(round(n_rows * validation))
    splits = np.full(n_rows, "test", dtype=object)
    splits[:n_train] = "train"
    splits[n_train : n_train + n_validation] = "validation"
    return splits


def group_expression(config: dict[str, Any]) -> str:
    """SQL giving each line its group: the adapter's group_id, or its time window."""
    grouping = config.get("grouping", {"method": "session"})
    method = grouping.get("method", "session")
    if method == "session":
        return "group_id"
    if method == "time_window":
        minutes = int(grouping["window_minutes"])
        if minutes <= 0:
            raise ValueError("grouping.window_minutes must be a positive number of minutes")
        # e.g. '2005-06-03 15:00' for every line logged from 15:00:00 to 15:59:59
        return f"strftime(time_bucket(INTERVAL {minutes} MINUTE, timestamp), '%Y-%m-%d %H:%M')"
    raise ValueError(f"grouping.method must be 'session' or 'time_window', got {method!r}")


def build_features(config: dict[str, Any]) -> FeatureStats:
    """Build the event-count matrix with time-based splits and save it as Parquet."""
    artifacts = Path(config["paths"]["artifacts_dir"])
    events_path = artifacts / EVENTS_FILE
    if not events_path.is_file():
        raise FileNotFoundError(f"{events_path} not found. Run `logad parse` first.")
    split_cfg = config.get("features", {}).get("split", {"train": 0.6, "validation": 0.2})

    # Which group does each line belong to? A session (HDFS block ID from the adapter) or a
    # fixed, non-overlapping time window (BGL: no sessions).
    con = duckdb.connect()
    con.sql(f"""
        CREATE VIEW grouped AS
        SELECT {group_expression(config)} AS group_id, line_id, timestamp, event_id, label
        FROM '{events_path.as_posix()}'
    """)

    # DuckDB counts events per group (11M rows → a few million (group, event) pairs) ...
    counts = con.sql("""
        SELECT group_id, event_id, count(*) AS n
        FROM grouped WHERE group_id IS NOT NULL
        GROUP BY group_id, event_id
    """).df()
    # A group is anomalous if any of its lines is (HDFS labels are per block anyway).
    blocks = con.sql("""
        SELECT group_id, min(timestamp) AS first_seen, min(line_id) AS first_line,
               max(label) AS label
        FROM grouped WHERE group_id IS NOT NULL
        GROUP BY group_id
    """).df()

    # ... and pandas turns the long table into a wide one: one column per event.
    matrix = counts.pivot_table(
        index="group_id", columns="event_id", values="n", fill_value=0, aggfunc="sum"
    )
    matrix = matrix[event_columns(matrix.columns)].astype("int32")

    # Oldest block first (ties broken by position in the log), then split by time.
    df = blocks.merge(matrix, left_on="group_id", right_index=True, validate="one_to_one")
    df = df.sort_values(["first_seen", "first_line"]).reset_index(drop=True)
    df["split"] = assign_splits(len(df), split_cfg["train"], split_cfg["validation"])
    df["label"] = df["label"].astype("Int8")
    df = df[META_COLUMNS + event_columns(df.columns)]

    part = artifacts / (FEATURES_FILE + ".part")
    df.to_parquet(part, index=False, compression="zstd")
    part.replace(artifacts / FEATURES_FILE)

    stats = FeatureStats(blocks=len(df), events=len(event_columns(df.columns)))
    stats.split_summary = summarize_splits(df)
    write_features_report(stats, config)
    return stats


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Blocks, anomalies and time range per split (labels are only COUNTED here)."""
    summary = (
        df.groupby("split", sort=False)
        .agg(
            blocks=("group_id", "size"),
            # all-null labels (unlabelled data) → unknown, not "0 anomalies"
            anomalies=("label", lambda s: s.sum() if s.notna().any() else pd.NA),
            start=("first_seen", "min"),
            end=("first_seen", "max"),
        )
        .reindex([s for s in SPLITS if s in set(df["split"])])
    )
    # Int64: the labels are 8-bit, and 100 × 27 would overflow an 8-bit integer.
    summary["anomalies"] = summary["anomalies"].astype("Int64")
    summary["anomaly_pct"] = (100 * summary["anomalies"] / summary["blocks"]).round(2)
    return summary


def load_features(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Load `features.parquet`; returns the table and the list of event columns."""
    path = Path(config["paths"]["artifacts_dir"]) / FEATURES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `logad features` first.")
    df = pd.read_parquet(path)
    return df, event_columns(df.columns)


def transformer_from_config(config: dict[str, Any]):
    """The (unfitted) count transformer from the config: fit it on the train split only."""
    features = config.get("features", {})
    return make_transformer(features.get("type", "event_count"), features.get("scale", False))


def write_features_report(stats: FeatureStats, config: dict[str, Any]) -> Path:
    reports = Path(config["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "features.md"
    grouping = config.get("grouping", {"method": "session"})
    unit = (
        f"{grouping['window_minutes']}-minute windows"
        if grouping.get("method") == "time_window"
        else "blocks"
    )
    lines = [
        f"# Features: {config['dataset']} ({Path(config['paths']['raw_dir']).name})",
        "",
        f"Generated by `logad features`. One row per {unit[:-1]}, one column per event (counts).",
        "",
        f"- Matrix: **{stats.blocks:,} {unit} × {stats.events} events**",
        f"- Split by time ({unit[:-1]}'s first log line): "
        "oldest → train, then validation, then test",
        "",
        f"| Split | {unit.capitalize()} | Anomalies | Anomaly % | From | To |",
        "|-------|-------:|----------:|----------:|------|----|",
    ]
    for split, row in stats.split_summary.iterrows():
        anomalies = "n/a" if pd.isna(row["anomalies"]) else f"{int(row['anomalies']):,}"
        pct = "n/a" if pd.isna(row["anomaly_pct"]) else f"{row['anomaly_pct']:.2f}"
        lines.append(
            f"| {split} | {int(row['blocks']):,} | {anomalies} | {pct} "
            f"| {row['start']:%Y-%m-%d %H:%M} | {row['end']:%Y-%m-%d %H:%M} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
