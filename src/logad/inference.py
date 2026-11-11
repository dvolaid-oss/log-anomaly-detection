"""Score NEW raw logs with a saved model bundle (used by the dashboard).

    raw lines → the dataset's adapter → saved Drain parser (match only, no learning)
              → event counts per group (same columns as in training)
              → score → alert if score >= threshold (chosen on validation)
              → reasons (explain module)

A bundle (`app/model/<dataset>/`) holds only small files this project created: the fitted
detector, the Drain parser, templates, the threshold and per-event training statistics.
It is written by `logad export-app`, so the dashboard runs without the big datasets.

Which model? The one with the best F1 on the VALIDATION split (never the test split). A
PyTorch autoencoder is converted to NumPy, so the online app never needs PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from logad.adapters import auth_log, hdfs
from logad.explain import Explainer, Reason
from logad.models.base import AnomalyDetector
from logad.parsing.drain import LogParser

BUNDLE_MODEL = "model.joblib"
BUNDLE_PARSER = "drain_parser.json.zlib"
BUNDLE_INFO = "bundle.json"
UNKNOWN = "unknown"
GROUP_NAMES = {"session": "block", "session_window": "IP window", "time_window": "window"}


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
        return Explainer(
            self.detector,
            self.events,
            self.templates,
            self.train_share,
            self.info.get("train_p99"),
        )

    @property
    def group_name(self) -> str:
        return GROUP_NAMES.get(self.info["grouping"].get("method", "session"), "group")


def best_model(thresholds: dict) -> str:
    """The model with the highest F1 on the validation split."""
    return max(thresholds, key=lambda m: thresholds[m]["validation_f1"])


def export_bundle(config: dict[str, Any], out_dir: Path, model: str | None = None) -> Path:
    """Write the dashboard bundle for one dataset (default model: best on validation)."""
    artifacts = Path(config["paths"]["artifacts_dir"])
    needed = {
        "parser": artifacts / BUNDLE_PARSER,
        "thresholds": artifacts / "thresholds.json",
        "templates": artifacts / "templates.csv",
        "features": artifacts / "features.parquet",
    }
    for name, path in needed.items():
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found ({name}). Run `logad run-all` first.")
    thresholds = json.loads(needed["thresholds"].read_text(encoding="utf-8"))
    model = model or best_model(thresholds)

    from logad.features import event_columns  # local import: avoids a cycle at import time

    features = pd.read_parquet(needed["features"])
    events = event_columns(features.columns)
    train = features.loc[features["split"] == "train", events]
    # keep_default_na=False: an empty template (BGL lines without a message) stays ""
    templates = pd.read_csv(needed["templates"], keep_default_na=False)

    detector = AnomalyDetector.load(artifacts / "models" / f"{model}.joblib")
    if model == "autoencoder":  # same weights, plain NumPy: no PyTorch needed to run it
        from logad.models.numpy_autoencoder import NumpyAutoencoder

        detector = NumpyAutoencoder.from_torch(detector)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, out_dir / BUNDLE_MODEL)
    (out_dir / BUNDLE_PARSER).write_bytes(needed["parser"].read_bytes())
    info = {
        "dataset": config["dataset"],
        "model": model,
        "threshold": thresholds[model]["threshold"],
        "validation_f1": thresholds[model]["validation_f1"],
        "grouping": config.get("grouping", {"method": "session"}),
        "adapter_options": config.get("data", {}).get("adapter_options") or {},
        "events": events,
        "templates": dict(zip(templates["event_id"], templates["template"], strict=True)),
        "train_share": (train > 0).mean().round(8).tolist(),
        "train_p99": train.quantile(0.99).round(2).tolist(),
        "train_groups": len(train),
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


# ----------------------------------------------------------------------- raw lines → groups
def _parser_for(info: dict[str, Any]):
    dataset = info["dataset"]
    if dataset == "hdfs":
        return hdfs.parse_lines
    if dataset == "auth":
        return partial(auth_log.parse_lines, **info.get("adapter_options", {}))
    raise ValueError(f"The dashboard doesn't support the {dataset!r} dataset yet.")


def assign_groups(lines: pd.DataFrame, grouping: dict[str, Any]) -> pd.Series:
    """Same groups as the features step (DuckDB time_bucket aligns like pandas floor)."""
    method = grouping.get("method", "session")
    if method == "session":
        return lines["group_id"]
    window = lines["timestamp"].dt.floor(f"{int(grouping['window_minutes'])}min")
    window = window.dt.strftime("%Y-%m-%d %H:%M")
    if method == "time_window":
        return window
    return lines["group_id"] + "@" + window  # session_window: e.g. '203.0.113.7@2026-05-14 02:10'


EXAMPLES = {
    "hdfs": "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block ...",
    "auth": "Mar  2 14:02:31 web01 sshd[2345]: Failed password for root from 203.0.113.7 ...",
}


@dataclass
class Analysis:
    blocks: pd.DataFrame  # one row per group: group_id, n_lines, score, alert, unknown_lines
    counts: pd.DataFrame  # event counts per group (index group_id, columns = bundle events)
    lines: pd.DataFrame  # parsed lines with their event_id (for showing a group's log)
    skipped_lines: int
    total_lines: int


def analyze_lines(raw_lines: list[str], bundle: ModelBundle) -> Analysis:
    """Parse, group, count, score and flag every group in a list of raw log lines."""
    lines, skipped = _parser_for(bundle.info)(raw_lines)
    lines = lines.assign(group_id=assign_groups(lines, bundle.info["grouping"]))
    lines = lines.dropna(subset=["group_id"])
    if lines.empty:
        example = EXAMPLES.get(bundle.info["dataset"], "")
        raise ValueError(
            f"No usable {bundle.info['dataset'].upper()} log lines found. This tab only "
            f"understands lines like: {example}"
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
    """The group's own log lines, in file order."""
    rows = analysis.lines[analysis.lines["group_id"] == group_id]
    return rows[["line_id", "timestamp", "level", "component", "event_id", "message"]]


def expected_vs_found(analysis: Analysis, bundle: ModelBundle, group_id: str) -> pd.DataFrame:
    """Per event: count found in the group vs what the model expected (rebuilding models)."""
    counts = analysis.counts.loc[group_id].to_numpy().astype(float)
    table = pd.DataFrame(
        {
            "event": bundle.events,
            "template": [bundle.templates.get(e, "?") for e in bundle.events],
            "found": counts.astype(int),
        }
    )
    if not bundle.detector.rebuilds:
        return table[table["found"] > 0].sort_values("found", ascending=False, ignore_index=True)
    _, expected = bundle.detector.reconstruction(counts.reshape(1, -1))
    table["expected"] = expected[0].round(1)
    table["difference"] = (table["found"] - table["expected"]).round(1)
    shown = (table["found"] > 0) | (table["expected"] >= 0.5)
    return table[shown].sort_values("difference", key=abs, ascending=False, ignore_index=True)


# ----------------------------------------------------------------------- demo logs
def _copy_lines(raw_log: Path, wanted: set[int], out: Path) -> None:
    with (
        raw_log.open(encoding="utf-8", errors="replace") as src,
        out.open("w", encoding="utf-8", newline="\n") as dst,
    ):
        for line_id, line in enumerate(src):
            if line_id in wanted:
                dst.write(line)


def export_demo(config: dict[str, Any], out_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    """Write a small demo log from the TEST split (never trained on) + per-group labels."""
    from logad.features import FEATURES_FILE  # local import: avoids a cycle at import time

    artifacts = Path(config["paths"]["artifacts_dir"])
    raw_log = Path(config["paths"]["raw_dir"]) / config["data"]["log_file"]
    for path in (artifacts / FEATURES_FILE, artifacts / "events.parquet", raw_log):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `logad run-all` first.")
    dataset = config["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, labels_path = out_dir / f"{dataset}_demo.log", out_dir / f"{dataset}_demo_labels.csv"
    features = pd.read_parquet(artifacts / FEATURES_FILE, columns=["group_id", "split", "label"])
    test = features[features["split"] == "test"]

    if dataset == "hdfs":
        # 270 normal + 30 anomalous blocks drawn at random from the test split
        picked = pd.concat(
            [
                test[test["label"] == 0].sample(270, random_state=seed),
                test[test["label"] == 1].sample(30, random_state=seed),
            ]
        )
        events = pd.read_parquet(artifacts / "events.parquet", columns=["line_id", "group_id"])
        _copy_lines(
            raw_log,
            set(events.loc[events["group_id"].isin(picked["group_id"]), "line_id"]),
            log_path,
        )  # noqa: E501
        labels = picked.assign(Label=picked["label"].map({0: "Normal", 1: "Anomaly"}))
        labels[["group_id", "Label"]].sort_values("group_id").to_csv(labels_path, index=False)
    elif dataset == "auth":
        _export_auth_demo(config, test, raw_log, log_path, labels_path)
    else:
        raise ValueError(f"No demo export for the {dataset!r} dataset.")
    return log_path, labels_path


def _export_auth_demo(config, test, raw_log, log_path, labels_path, days: int = 2) -> None:
    """A continuous slice of the TEST period (default 2 days): the one containing the most
    attack types, so the demo shows variety. Labels name the attack type(s) per IP window."""
    events = pd.read_parquet(
        Path(config["paths"]["artifacts_dir"]) / "events.parquet",
        columns=["line_id", "timestamp"],
    )
    attacks = pd.read_csv(
        Path(config["paths"]["raw_dir"]) / config["data"]["label_file"], keep_default_na=False
    )
    lines = events.merge(attacks, on="line_id")
    test_start = pd.to_datetime(test["group_id"].str.split("@").str[1]).min()
    lines = lines[lines["timestamp"] >= test_start]

    best_start, best_types = None, -1
    for start in pd.date_range(test_start.normalize(), lines["timestamp"].max(), freq="D"):
        window = lines[
            (lines["timestamp"] >= start) & (lines["timestamp"] < start + pd.Timedelta(days=days))
        ]  # noqa: E501
        n_types = window.loc[window["attack"] != "", "attack"].nunique()
        if n_types > best_types:
            best_start, best_types = start, n_types
    end = best_start + pd.Timedelta(days=days)
    chosen = lines[(lines["timestamp"] >= best_start) & (lines["timestamp"] < end)]
    _copy_lines(raw_log, set(chosen["line_id"]), log_path)

    # Per-group labels, computed exactly like the dashboard will group the demo lines.
    parsed, _ = auth_log.parse_lines(
        log_path.read_text(encoding="utf-8").splitlines(),
        **(config["data"].get("adapter_options") or {}),
    )
    if len(parsed) != len(chosen):  # every demo line must parse, or labels would shift
        raise ValueError(f"Demo parsing mismatch: {len(parsed)} parsed vs {len(chosen)} lines")
    parsed["attack"] = chosen["attack"].to_numpy()
    parsed["group_id"] = assign_groups(parsed, config["grouping"])
    per_group = (
        parsed.dropna(subset=["group_id"])
        .groupby("group_id")["attack"]
        .agg(lambda a: ";".join(sorted(set(a) - {""})))
        .reset_index()
    )
    per_group["Label"] = per_group["attack"].map(lambda a: "Anomaly" if a else "Normal")
    per_group[["group_id", "Label", "attack"]].to_csv(labels_path, index=False)
