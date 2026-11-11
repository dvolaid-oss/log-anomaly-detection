"""auth.log adapter: syslog-style SSH login lines → the common schema.

    Mar  2 00:02:23 web01 sshd[8110]: Failed password for root from 192.0.2.74 port 3280 ssh2
    └─timestamp───┘ └host┘ └process┘  └──────────────────message──────────────────────────┘

- `group_id` = the SOURCE IP in the message ("from 192.0.2.74", "by 192.0.2.74", ...): attacks
  are IP-centric, and the features step groups by IP + time window. Lines without an IP
  (e.g. "session opened for user alice") keep an empty group and are not scored.
- syslog lines have no year, so it comes from the config (`data.adapter_options.year`).
- `label` comes from the generator's labels file (line_id → attack type), evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from logad.adapters.base import COLUMNS, to_arrow

SOURCE = "auth"
LINE_PATTERN = (
    r"^(?P<stamp>\w{3} +\d{1,2} \d\d:\d\d:\d\d) (?P<host>\S+) "
    r"(?P<process>[\w.-]+)(?:\[\d+\])?: (?P<message>.*)$"
)
IP_PATTERN = r"(\d{1,3}(?:\.\d{1,3}){3})"


@dataclass
class AuthIngestStats:
    lines: int = 0
    skipped_lines: int = 0
    rows: int = 0
    lines_without_ip: int = 0
    source_ips: int = 0
    attack_lines: int = 0


def parse_lines(lines: list[str], first_line_id: int = 0, year: int = 2026):
    """Parse raw auth.log lines into common-schema rows. Returns (rows, skipped count)."""
    raw = pd.Series(lines, dtype="str").str.rstrip("\r\n")
    parts = raw.str.extract(LINE_PATTERN)
    parts["line_id"] = range(first_line_id, first_line_id + len(lines))
    malformed = parts["message"].isna()
    parts = parts[~malformed]

    stamps = str(year) + " " + parts["stamp"].str.replace(r" +", " ", regex=True)
    df = pd.DataFrame(
        {
            "line_id": parts["line_id"],
            "timestamp": pd.to_datetime(stamps, format="%Y %b %d %H:%M:%S"),
            "source": SOURCE,
            "level": "INFO",  # syslog auth lines carry no level
            "component": parts["process"],
            "message": parts["message"],
            "group_id": parts["message"].str.extract(IP_PATTERN, expand=False),
            "label": pd.NA,
        }
    )
    return df[COLUMNS].reset_index(drop=True), int(malformed.sum())


def load_labels(path: Path) -> pd.Series:
    """line_id → 1 (attack) / 0 (normal), from the generator's labels file."""
    labels = pd.read_csv(path, keep_default_na=False)
    return (labels.set_index("line_id")["attack"] != "").astype("int8")


def ingest(
    log_path: Path,
    out_path: Path,
    label_path: Path | None = None,
    chunk_size: int = 1_000_000,
    year: int = 2026,
) -> AuthIngestStats:
    labels = load_labels(label_path) if label_path else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    stats, ips, writer = AuthIngestStats(), set(), None
    try:
        with log_path.open(encoding="utf-8", errors="replace") as f:
            while batch := list(islice(f, chunk_size)):
                df, skipped = parse_lines(batch, first_line_id=stats.lines, year=year)
                if labels is not None:
                    df["label"] = df["line_id"].map(labels).astype("Int8")
                table = to_arrow(df)
                if writer is None:
                    writer = pq.ParquetWriter(part, table.schema, compression="zstd")
                writer.write_table(table)
                stats.lines += len(batch)
                stats.skipped_lines += skipped
                stats.rows += len(df)
                stats.lines_without_ip += int(df["group_id"].isna().sum())
                stats.attack_lines += int(df["label"].fillna(0).sum())
                ips.update(df["group_id"].dropna())
        if writer is None:
            raise ValueError(f"{log_path} is empty: nothing to ingest.")
    except BaseException:
        if writer is not None:
            writer.close()
        part.unlink(missing_ok=True)
        raise
    writer.close()
    part.replace(out_path)
    stats.source_ips = len(ips)
    return stats
