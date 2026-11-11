"""BGL adapter: raw LogHub BGL lines (BlueGene/L supercomputer) → the common schema.

A raw BGL line has 9 fixed fields, then the free-text message. For example:

    -                            LABEL: "-" = normal, else an alert category (e.g. KERNDTLB)
    1117838570                   unix time
    2005.06.03                   date
    R02-M1-N0-C:J12-U11          node
    2005-06-03-15.42.50.675872   time
    R02-M1-N0-C:J12-U11          node again
    RAS KERNEL INFO              type, component, level
    instruction cache parity error corrected      message

**Leakage trap:** the label is the first word of every raw line. The adapter moves it into
the `label` column and it never appears in `message`, so no model can read it (tested).

BGL has no sessions, so `group_id` stays empty here: lines are grouped into time windows
in the features step (`grouping.method: time_window`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from logad.adapters.base import COLUMNS, to_arrow

SOURCE = "bgl"
LINE_PATTERN = (
    r"^(?P<label>\S+) (?P<unix>\d+) (?P<date>\S+) (?P<node>\S+) (?P<time>\S+) "
    r"(?P<node2>\S+) (?P<type>\S+) (?P<component>\S+) (?P<level>\S+)(?: (?P<message>.*))?$"
)


@dataclass
class BglIngestStats:
    lines: int = 0
    skipped_lines: int = 0
    rows: int = 0
    anomalous_lines: int = 0  # lines whose label is an alert category (not "-")


def parse_lines(lines: list[str], first_line_id: int = 0) -> tuple[pd.DataFrame, int]:
    """Parse raw BGL lines into common-schema rows. Returns (rows, skipped line count)."""
    raw = pd.Series(lines, dtype="str").str.rstrip("\r\n")
    parts = raw.str.extract(LINE_PATTERN)
    parts["line_id"] = range(first_line_id, first_line_id + len(lines))
    malformed = parts["unix"].isna()
    parts = parts[~malformed]

    df = pd.DataFrame(
        {
            "line_id": parts["line_id"],
            # unix seconds (UTC); the 'time' field has microseconds we don't need
            "timestamp": pd.to_datetime(parts["unix"].astype("int64"), unit="s"),
            "source": SOURCE,
            "level": parts["level"],
            "component": parts["component"],
            "message": parts["message"].fillna(""),  # a few BGL lines have no message
            "group_id": None,  # time windows are built later, in the features step
            "label": (parts["label"] != "-").astype("int8"),  # 1 = alert line
        }
    )
    return df[COLUMNS].reset_index(drop=True), int(malformed.sum())


def ingest(
    log_path: Path,
    out_path: Path,
    label_path: Path | None = None,  # unused: BGL labels are inside each line
    chunk_size: int = 1_000_000,
) -> BglIngestStats:
    """Convert a raw BGL log into Parquet in the common schema, in chunks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    stats = BglIngestStats()
    writer = None
    try:
        # BGL has a few non-UTF-8 bytes; replace them instead of crashing.
        with log_path.open(encoding="utf-8", errors="replace") as f:
            while batch := list(islice(f, chunk_size)):
                df, skipped = parse_lines(batch, first_line_id=stats.lines)
                table = to_arrow(df)
                if writer is None:
                    writer = pq.ParquetWriter(part, table.schema, compression="zstd")
                writer.write_table(table)
                stats.lines += len(batch)
                stats.skipped_lines += skipped
                stats.rows += len(df)
                stats.anomalous_lines += int(df["label"].sum())
        if writer is None:
            raise ValueError(f"{log_path} is empty: nothing to ingest.")
    except BaseException:
        if writer is not None:
            writer.close()
        part.unlink(missing_ok=True)
        raise
    writer.close()
    part.replace(out_path)
    return stats
