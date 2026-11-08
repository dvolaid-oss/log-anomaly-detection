"""The common schema every adapter produces.

Each log source has its own format, but after its adapter runs, every dataset is a table
with exactly these columns. Everything downstream (parsing, features, models) only knows
this schema, so adding a new log type means writing one new adapter and nothing else.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

# One row per (log line, group). A line that mentions several groups (e.g. an HDFS
# "delete these 9 blocks" line) appears once per group, all with the same line_id.
SCHEMA = pa.schema(
    [
        ("line_id", pa.int64()),  # position of the line in the raw file (0-based)
        ("timestamp", pa.timestamp("s")),  # when it was logged (second precision)
        ("source", pa.string()),  # which dataset it came from, e.g. "hdfs"
        ("level", pa.string()),  # INFO, WARN, ERROR, ...
        ("component", pa.string()),  # the program part that logged it
        ("message", pa.string()),  # free text, later turned into a template by Drain
        ("group_id", pa.string()),  # session the line belongs to (HDFS: block ID)
        ("label", pa.int8()),  # 1 = anomaly, 0 = normal, null = unknown. EVALUATION ONLY.
    ]
)
COLUMNS = SCHEMA.names


def to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert an adapter's DataFrame to an Arrow table with the common schema.

    Raises if columns are missing or extra, so a buggy adapter fails loudly here
    instead of silently producing bad data for later stages.
    """
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Adapter produced columns {list(df.columns)}, expected {COLUMNS}")
    return pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
