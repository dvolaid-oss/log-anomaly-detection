"""Score NEW raw HDFS logs with a saved model bundle (used by the dashboard).

    raw lines → HDFS adapter → saved Drain parser (match only, no learning)
              → event counts per block (same columns as in training)
              → PCA score → alert if score >= threshold (chosen on validation)
              → reasons (explain module)

The bundle (`app/model/`) holds only small files this project created: the fitted PCA
detector, the Drain parser, templates, the threshold and per-event training statistics.
It is written by `logad export-app`, so the dashboard runs without the 1.6 GB dataset.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from logad.adapters.hdfs import parse_lines
from logad.explain import Explainer, Reason
from logad.models.base import AnomalyDetector
from logad.parsing.drain import LogParser

BUNDLE_MODEL = "pca.joblib"
BUNDLE_PARSER = "drain_parser.json.zlib"
BUNDLE_INFO = "bundle.json"
UNKNOWN = "unknown"


@dataclass
class ModelBundle:
    detector: AnomalyDetector
    parser: LogParser
    events: list[str]
    templates: dict[str, str]
    threshold: float
    train_share: list[float]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def explainer(self) -> Explainer:
        return Explainer(self.detector, self.events, self.templates, self.train_share)


def export_bundle(config: dict[str, Any], out_dir: Path, model: str = "pca") -> Path:
    """Copy the fitted model + parser and write templates, threshold and training stats."""
    artifacts = Path(config["paths"]["artifacts_dir"])
    needed = {
        "model": artifacts / "models" / f"{model}.joblib",
        "parser": artifacts / BUNDLE_PARSER,
        "thresholds": artifacts / "thresholds.json",
        "templates": artifacts / "templates.csv",
        "features": artifacts / "features.parquet",
    }
    for name, path in needed.items():
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found ({name}). Run `logad run-all` first.")

    from logad.features import event_columns  # local import: avoids a cycle at import time

    features = pd.read_parquet(needed["features"])
    events = event_columns(features.columns)
    train = features.loc[features["split"] == "train", events]
    thresholds = json.loads(needed["thresholds"].read_text(encoding="utf-8"))
    # keep_default_na=False: an empty template (BGL lines without a message) stays ""
    templates = pd.read_csv(needed["templates"], keep_default_na=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(needed["model"], out_dir / BUNDLE_MODEL)
    shutil.copyfile(needed["parser"], out_dir / BUNDLE_PARSER)
    info = {
        "dataset": config["dataset"],
        "model": model,
        "threshold": thresholds[model]["threshold"],
        "validation_f1": thresholds[model]["validation_f1"],
        "events": events,
        "templates": dict(zip(templates["event_id"], templates["template"], strict=True)),
        "train_share": (train > 0).mean().round(8).tolist(),
        "train_blocks": len(train),
    }
    (out_dir / BUNDLE_INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")
    return out_dir


def load_bundle(bundle_dir: Path) -> ModelBundle:
    """Load a bundle written by `export_bundle` (only load bundles this project made)."""
    info = json.loads((bundle_dir / BUNDLE_INFO).read_text(encoding="utf-8"))
    return ModelBundle(
        detector=AnomalyDetector.load(bundle_dir / BUNDLE_MODEL),
        parser=LogParser.load(bundle_dir / BUNDLE_PARSER),
        events=info["events"],
        templates=info["templates"],
        threshold=info["threshold"],
        train_share=info["train_share"],
        info=info,
    )


@dataclass
class Analysis:
    blocks: pd.DataFrame  # one row per block: group_id, n_lines, score, alert, unknown_lines
    counts: pd.DataFrame  # event counts per block (index group_id, columns = bundle events)
    lines: pd.DataFrame  # parsed lines with their event_id (for showing a block's log)
    skipped_lines: int
    total_lines: int


def analyze_lines(raw_lines: list[str], bundle: ModelBundle) -> Analysis:
    """Parse, count, score and flag every block in a list of raw HDFS log lines."""
    lines, skipped = parse_lines(raw_lines)
    lines = lines.dropna(subset=["group_id"])
    if lines.empty:
        raise ValueError(
            "No HDFS log lines with block IDs found. This app only understands HDFS logs "
            "(lines like '081109 203615 148 INFO dfs.DataNode$PacketResponder: ...')."
        )

    # Mask, then match each DISTINCT masked message once (same speed trick as `logad parse`;
    # masking is idempotent, so match() masking again changes nothing).
    masked = lines["message"].map(bundle.parser.mask)
    event_of = {m: bundle.parser.match(m) or UNKNOWN for m in pd.unique(masked)}
    lines = lines.assign(event_id=masked.map(event_of))

    counts = lines.groupby(["group_id", "event_id"]).size().unstack(fill_value=0).astype("int32")
    unknown = counts.pop(UNKNOWN) if UNKNOWN in counts else pd.Series(0, index=counts.index)
    counts = counts.reindex(columns=bundle.events, fill_value=0)

    scores = bundle.detector.score(counts.to_numpy())
    blocks = pd.DataFrame(
        {
            "group_id": counts.index,
            "n_lines": counts.sum(axis=1).to_numpy() + unknown.to_numpy(),
            "score": scores,
            "alert": scores >= bundle.threshold,
            "unknown_lines": unknown.to_numpy(),
        }
    ).sort_values("score", ascending=False, ignore_index=True)
    return Analysis(blocks, counts, lines, skipped, len(raw_lines))


def explain_block(analysis: Analysis, bundle: ModelBundle, group_id: str) -> list[Reason]:
    row = analysis.blocks.set_index("group_id").loc[group_id]
    counts = analysis.counts.loc[group_id].to_numpy()
    return bundle.explainer.explain(counts, unknown_lines=int(row["unknown_lines"]))


def block_lines(analysis: Analysis, group_id: str) -> pd.DataFrame:
    """The block's own log lines, in file order."""
    rows = analysis.lines[analysis.lines["group_id"] == group_id]
    return rows[["line_id", "timestamp", "level", "component", "event_id", "message"]]


def expected_vs_found(analysis: Analysis, bundle: ModelBundle, group_id: str) -> pd.DataFrame:
    """Per event: count found in the block vs what PCA expected (for the detail view)."""
    counts = analysis.counts.loc[group_id].to_numpy().astype(float)
    explainer = bundle.explainer
    x = bundle.detector.transformer.transform(counts.reshape(1, -1))
    pca = bundle.detector.pca_
    expected = np.maximum(pca.inverse_transform(pca.transform(x))[0] / explainer.idf, 0)
    table = pd.DataFrame(
        {
            "event": bundle.events,
            "template": [bundle.templates.get(e, "?") for e in bundle.events],
            "found": counts.astype(int),
            "expected": expected.round(1),
        }
    )
    table["difference"] = (table["found"] - table["expected"]).round(1)
    shown = (table["found"] > 0) | (table["expected"] >= 0.5)
    return table[shown].sort_values("difference", key=abs, ascending=False, ignore_index=True)


def export_demo(
    config: dict[str, Any],
    out_dir: Path,
    n_normal: int = 270,
    n_anomalous: int = 30,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write a small demo log of TEST-split blocks (never seen in training) + their labels.

    Blocks are drawn at random (fixed seed) from the test split, so the demo shows the
    model's honest behaviour, misses included. The raw lines are copied unchanged.
    """
    from logad.features import FEATURES_FILE  # local import: avoids a cycle at import time

    artifacts = Path(config["paths"]["artifacts_dir"])
    raw_log = Path(config["paths"]["raw_dir"]) / config["data"]["log_file"]
    for path in (artifacts / FEATURES_FILE, artifacts / "events.parquet", raw_log):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `logad run-all` first.")

    blocks = pd.read_parquet(artifacts / FEATURES_FILE, columns=["group_id", "split", "label"])
    test = blocks[blocks["split"] == "test"]
    picked = pd.concat(
        [
            test[test["label"] == 0].sample(n_normal, random_state=seed),
            test[test["label"] == 1].sample(n_anomalous, random_state=seed),
        ]
    )
    events = pd.read_parquet(artifacts / "events.parquet", columns=["line_id", "group_id"])
    wanted = set(events.loc[events["group_id"].isin(picked["group_id"]), "line_id"])

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, labels_path = out_dir / "hdfs_demo.log", out_dir / "hdfs_demo_labels.csv"
    with (
        raw_log.open(encoding="utf-8", errors="replace") as src,
        log_path.open("w", encoding="utf-8", newline="\n") as dst,
    ):
        for line_id, line in enumerate(src):
            if line_id in wanted:
                dst.write(line)
    labels = picked.rename(columns={"group_id": "BlockId"})
    labels["Label"] = labels["label"].map({0: "Normal", 1: "Anomaly"})
    labels[["BlockId", "Label"]].sort_values("BlockId").to_csv(labels_path, index=False)
    return log_path, labels_path
