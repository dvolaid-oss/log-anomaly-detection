"""HDFS adapter: raw LogHub HDFS lines → the common schema.

A raw HDFS line looks like this:

    081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_388 ...
    └date┘ └time┘ └pid┘└lvl┘ └──────component──────────┘ └──────────message──────────────────┘

Each line is about one or more HDFS *blocks* (`blk_<number>`). The block ID is the
session ("group") a line belongs to, and the labels file says whether each block is
normal or anomalous.

The full dataset has ~11M lines (1.5 GB), so the file is processed in chunks: only one
chunk is in memory at a time, and each chunk is appended to the output Parquet file.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from logad.adapters.base import COLUMNS, to_arrow

SOURCE = "hdfs"
LINE_PATTERN = (
    r"^(?P<date>\d{6}) (?P<time>\d{6}) (?P<pid>\d+) (?P<level>[A-Z]+) "
    r"(?P<component>[^:]+): (?P<message>.*)$"
)
BLOCK_PATTERN = r"blk_-?\d+"  # block IDs can be negative: blk_-6952295868487656571
LABEL_VALUES = {"Normal": 0, "Anomaly": 1}


@dataclass
class IngestStats:
    """Numbers reported after ingesting, to sanity-check the result."""

    lines: int = 0  # raw lines read
    skipped_lines: int = 0  # lines that did not match the HDFS format
    rows: int = 0  # rows written (a multi-block line gives several rows)
    lines_without_group: int = 0  # lines with no block ID
    groups: int = 0  # distinct block IDs
    anomalous_groups: int = 0  # blocks labelled "Anomaly"
    unlabelled_groups: int = 0  # blocks missing from the labels file


def parse_lines(lines: list[str], first_line_id: int = 0) -> tuple[pd.DataFrame, int]:
    """Parse raw HDFS lines into common-schema rows (label still empty).

    Returns the DataFrame and how many lines were skipped because they did not match
    the expected format.
    """
    raw = pd.Series(lines, dtype="str").str.rstrip("\r\n")
    parts = raw.str.extract(LINE_PATTERN)
    parts["line_id"] = range(first_line_id, first_line_id + len(lines))

    malformed = parts["message"].isna()
    parts = parts[~malformed]

    df = pd.DataFrame(
        {
            "line_id": parts["line_id"],
            "timestamp": pd.to_datetime(
                parts["date"] + " " + parts["time"], format="%y%m%d %H%M%S"
            ),
            "source": SOURCE,
            "level": parts["level"],
            "component": parts["component"],
            "message": parts["message"],
            # Every block ID in the line, as a list. Most lines have exactly one.
            "group_id": parts["message"].str.findall(BLOCK_PATTERN),
            "label": pd.NA,
        }
    )
    # One row per (line, block). A line with no block keeps one row with group_id = NaN.
    df = df.explode("group_id").drop_duplicates(["line_id", "group_id"])
    return df[COLUMNS].reset_index(drop=True), int(malformed.sum())


def load_labels(path: Path) -> dict[str, int]:
    """Read the LogHub labels file (columns BlockId, Label) → {block_id: 0 or 1}."""
    labels = pd.read_csv(path, dtype="str")
    if list(labels.columns) != ["BlockId", "Label"]:
        raise ValueError(f"Expected columns BlockId,Label in {path}, got {list(labels.columns)}")
    unknown = set(labels["Label"]) - LABEL_VALUES.keys()
    if unknown:
        raise ValueError(f"Unknown label values in {path}: {sorted(unknown)}")
    return dict(zip(labels["BlockId"], labels["Label"].map(LABEL_VALUES), strict=True))


def ingest(
    log_path: Path,
    out_path: Path,
    label_path: Path | None = None,
    chunk_size: int = 1_000_000,
) -> IngestStats:
    """Convert a raw HDFS log file into a Parquet file in the common schema.

    Labels are attached for evaluation only; no training step ever reads them.
    The output is written to `<out_path>.part` and renamed at the end, so a crash
    never leaves a half-written file that looks finished.
    """
    labels = load_labels(label_path) if label_path else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")

    stats = IngestStats()
    seen_groups: set[str] = set()
    writer = None
    try:
        with log_path.open(encoding="utf-8", errors="replace") as f:
            while batch := list(islice(f, chunk_size)):
                df, skipped = parse_lines(batch, first_line_id=stats.lines)
                df["label"] = df["group_id"].map(labels).astype("Int8")

                table = to_arrow(df)
                if writer is None:
                    writer = pq.ParquetWriter(part, table.schema, compression="zstd")
                writer.write_table(table)

                stats.lines += len(batch)
                stats.skipped_lines += skipped
                stats.rows += len(df)
                stats.lines_without_group += int(df["group_id"].isna().sum())
                seen_groups.update(df["group_id"].dropna())
        if writer is None:
            raise ValueError(f"{log_path} is empty: nothing to ingest.")
    except BaseException:
        if writer is not None:
            writer.close()
        part.unlink(missing_ok=True)
        raise
    writer.close()
    part.replace(out_path)

    stats.groups = len(seen_groups)
    stats.anomalous_groups = sum(labels.get(g) == 1 for g in seen_groups)
    stats.unlabelled_groups = sum(g not in labels for g in seen_groups)
    return stats
