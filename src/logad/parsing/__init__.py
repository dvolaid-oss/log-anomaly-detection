"""Log parsing: `logs.parquet` → event IDs + templates, using masking and Drain.

`parse_dataset(config)` runs the whole step:

1. DuckDB masks every message (block IDs, IPs, numbers → placeholders) straight from the
   Parquet file, and keeps only the DISTINCT masked messages, in order of first appearance.
   11.2M HDFS lines shrink to a few hundred distinct masked messages.
2. Drain learns templates from those distinct messages (fast, even in pure Python).
3. DuckDB joins each line back to its event ID and writes `events.parquet`.

Why the shortcut is safe: Drain only ever looks at the masked text, so identical masked
messages always land in the same group. Feeding each one once instead of thousands of times
gives the same groups, minutes faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from logad.adapters import OUTPUT_FILE as LOGS_FILE
from logad.parsing.accuracy import grouping_accuracy
from logad.parsing.drain import LogParser, Mask

EVENTS_FILE = "events.parquet"
TEMPLATES_FILE = "templates.csv"
PARSER_FILE = "drain_parser.json.zlib"


@dataclass
class ParseStats:
    lines: int = 0
    distinct_messages: int = 0
    distinct_masked: int = 0  # what Drain actually had to read
    templates: int = 0
    grouping_accuracy: float | None = None  # only when ground truth is available
    true_templates: int | None = None


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    """Run a query that returns one value (DuckDB 2.0 relations have no fetchone())."""
    return con.sql(sql).fetchall()[0][0]


def sql_mask_expression(column: str, masks: list[Mask]) -> str:
    """Build nested DuckDB regexp_replace(...) calls applying the masks in order.

    Must give the same result as LogParser.mask (tested), since DuckDB masks the bulk data
    and Python masks single new messages later.
    """
    expression = column
    for mask in masks:
        pattern = mask.pattern.replace("'", "''")  # escape quotes for SQL
        expression = f"regexp_replace({expression}, '{pattern}', '{mask.placeholder}', 'g')"
    return expression


def parse_dataset(config: dict[str, Any]) -> ParseStats:
    """Run masking + Drain on `<artifacts_dir>/logs.parquet` and save the results."""
    artifacts = Path(config["paths"]["artifacts_dir"])
    logs_path = artifacts / LOGS_FILE
    if not logs_path.is_file():
        raise FileNotFoundError(f"{logs_path} not found. Run `logad ingest` first.")

    parser = LogParser.from_config(config)
    masked = sql_mask_expression("message", parser.masks)
    con = duckdb.connect()
    con.sql(f"CREATE VIEW logs AS SELECT * FROM '{logs_path.as_posix()}'")

    # One row per raw line (multi-block lines appear once per block in logs.parquet).
    con.sql(f"""
        CREATE TABLE lines AS
        SELECT DISTINCT ON (line_id) line_id, message, {masked} AS masked
        FROM logs
    """)
    stats = ParseStats(
        lines=_scalar(con, "SELECT count(*) FROM lines"),
        distinct_messages=_scalar(con, "SELECT count(DISTINCT message) FROM lines"),
    )

    # Distinct masked messages in order of first appearance, so Drain sees them in the
    # same order as it would when reading the log top to bottom.
    distinct = con.sql("""
        SELECT masked, min(line_id) AS first_line FROM lines
        GROUP BY masked ORDER BY first_line
    """).df()
    distinct["event_id"] = parser.fit(distinct["masked"])
    stats.distinct_masked = len(distinct)

    templates = pd.DataFrame(parser.templates.items(), columns=["event_id", "template"])
    stats.templates = len(templates)

    # Join every (line, block) row back to its event ID and save.
    con.register("mapping", distinct[["masked", "event_id"]])
    con.sql("""
        CREATE TABLE line_events AS
        SELECT l.line_id, m.event_id FROM lines l JOIN mapping m USING (masked)
    """)
    events_path = artifacts / EVENTS_FILE
    part = events_path.with_name(EVENTS_FILE + ".part")  # renamed when complete
    con.sql(f"""
        COPY (
            SELECT g.line_id, g.timestamp, g.group_id, e.event_id, g.label
            FROM logs g JOIN line_events e USING (line_id)
            ORDER BY g.line_id, g.group_id
        ) TO '{part.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    part.replace(events_path)

    counts = con.sql("""
        SELECT event_id, count(*) AS lines FROM line_events GROUP BY event_id
    """).df()
    templates = templates.merge(counts, on="event_id").sort_values("lines", ascending=False)
    templates["pct"] = (100 * templates["lines"] / stats.lines).round(4)
    templates.to_csv(artifacts / TEMPLATES_FILE, index=False)
    parser.save(artifacts / PARSER_FILE)
    write_templates_report(templates, config)

    truth_file = config.get("data", {}).get("truth_file")
    if truth_file:
        truth_path = Path(config["paths"]["raw_dir"]) / truth_file
        predicted = con.sql("SELECT line_id, event_id FROM line_events ORDER BY line_id").df()
        stats.grouping_accuracy, stats.true_templates = score_against_truth(truth_path, predicted)
    return stats


def score_against_truth(truth_path: Path, predicted: pd.DataFrame) -> tuple[float, int]:
    """Compare with a LogHub `*_structured.csv` (LineId is 1-based, our line_id 0-based)."""
    if not truth_path.is_file():
        raise FileNotFoundError(f"{truth_path} not found. Run `logad download` first.")
    truth = pd.read_csv(truth_path, usecols=["LineId", "EventId"])
    truth["line_id"] = truth["LineId"] - 1
    merged = truth.merge(predicted, on="line_id", how="inner", validate="one_to_one")
    if len(merged) != len(truth):
        raise ValueError(f"Only {len(merged)} of {len(truth)} ground-truth lines matched")
    return grouping_accuracy(merged["EventId"], merged["event_id"]), truth["EventId"].nunique()


def write_templates_report(templates: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Markdown table of all templates, committed to the repo so it shows on GitHub."""
    reports = Path(config["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "templates.md"
    lines = [
        f"# Log templates: {config['dataset']} ({Path(config['paths']['raw_dir']).name})",
        "",
        "Generated by `logad parse` (masking + Drain). Sorted by frequency.",
        "",
        "| Event | Template | Lines | % |",
        "|-------|----------|------:|--:|",
    ]
    for row in templates.itertuples():
        template = row.template.replace("|", "\\|")
        lines.append(f"| {row.event_id} | `{template}` | {row.lines:,} | {row.pct:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
