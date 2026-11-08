"""Load and validate YAML config files.

Every stage reads its settings (paths, window sizes, hyperparameters) from a file in
`configs/`, so nothing is hardcoded and the same code runs on any dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Keys every config must have. Stage-specific keys are checked by the stage that uses them.
REQUIRED_KEYS = {"dataset", "seed", "paths"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and check that it has the required top-level keys."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        # safe_load only builds plain data (dicts, lists, strings, numbers);
        # yaml.load could run arbitrary Python objects from a malicious file.
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping of key: value pairs: {path}")

    missing = REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required keys: {sorted(missing)}")

    return config
