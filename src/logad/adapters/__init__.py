"""Dataset adapters: convert each log source (HDFS, BGL, auth.log, ...) to one common schema.

`ingest_dataset(config)` picks the adapter from the config's `dataset` key, so the CLI
never needs to know which log format it is dealing with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from logad.adapters import hdfs

ADAPTERS = {"hdfs": hdfs.ingest}
OUTPUT_FILE = "logs.parquet"


def ingest_dataset(config: dict[str, Any]) -> tuple[Path, hdfs.IngestStats]:
    """Run the dataset's adapter: raw log file → `<artifacts_dir>/logs.parquet`."""
    dataset = config["dataset"]
    if dataset not in ADAPTERS:
        raise ValueError(f"No adapter for dataset {dataset!r} yet. Available: {sorted(ADAPTERS)}")

    raw_dir = Path(config["paths"]["raw_dir"])
    data = config.get("data", {})
    if "log_file" not in data:
        raise ValueError("Config has no 'data.log_file' entry.")

    log_path = raw_dir / data["log_file"]
    if not log_path.is_file():
        raise FileNotFoundError(f"{log_path} not found. Run `logad download` first.")
    label_path = raw_dir / data["label_file"] if data.get("label_file") else None
    if label_path and not label_path.is_file():
        raise FileNotFoundError(f"Labels file {label_path} not found. Run `logad download`.")

    out_path = Path(config["paths"]["artifacts_dir"]) / OUTPUT_FILE
    stats = ADAPTERS[dataset](log_path, out_path, label_path)
    return out_path, stats
