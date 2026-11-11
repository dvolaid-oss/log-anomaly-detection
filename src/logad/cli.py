"""Command-line interface for the pipeline.

Usage:
    logad show-config --config configs/hdfs.yaml
    logad download --config configs/hdfs.yaml
    logad ingest --config configs/hdfs.yaml
    logad parse --config configs/hdfs.yaml
    logad features --config configs/hdfs.yaml
    logad train --config configs/hdfs.yaml
    logad evaluate --config configs/hdfs.yaml
    logad run-all --config configs/hdfs.yaml     # all of the above, in order

Each pipeline step is one command, and each reads its settings from the config file.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from logad import __version__
from logad.adapters import ingest_dataset
from logad.config import load_config
from logad.download import download_dataset
from logad.evaluation import evaluate_scores
from logad.features import build_features
from logad.inference import export_bundle, export_demo
from logad.models import train_models
from logad.parsing import parse_dataset

app = typer.Typer(
    help="Log anomaly detection pipeline: download → ingest → parse → features → train → evaluate.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_CONFIG = Path("configs/hdfs.yaml")
ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config file.")]


def friendly_errors(command: Callable) -> Callable:
    """Show expected problems (missing files, bad config) as one clear line, not a traceback."""

    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except (FileNotFoundError, ValueError) as error:
            typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from error

    return wrapper


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"logad {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Log anomaly detection pipeline: download → ingest → parse → features → train → evaluate."""


@app.command("show-config")
@friendly_errors
def show_config(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Print the loaded config, to check which settings a run will use."""
    cfg = load_config(config)
    typer.echo(f"dataset: {cfg['dataset']}")
    typer.echo(f"seed:    {cfg['seed']}")
    for name, value in cfg["paths"].items():
        typer.echo(f"paths.{name}: {value}")


@app.command()
@friendly_errors
def download(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Download the raw dataset from official LogHub sources and verify checksums."""
    cfg = load_config(config)
    download_dataset(cfg, report=typer.echo)
    typer.echo(f"[{cfg['dataset']}] Raw data is ready in {cfg['paths']['raw_dir']}")


@app.command()
@friendly_errors
def ingest(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Convert raw logs to the common schema and save them as Parquet."""
    cfg = load_config(config)
    typer.echo(
        f"[{cfg['dataset']}] Ingesting {cfg['paths']['raw_dir']} (may take a few minutes)..."
    )
    out_path, stats = ingest_dataset(cfg)
    for name, value in dataclasses.asdict(stats).items():  # each adapter reports its own stats
        typer.echo(f"  {name.replace('_', ' ') + ':':<22}{value:,}")
    typer.echo(f"Saved {out_path}")


@app.command()
@friendly_errors
def parse(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Turn log messages into templates (event IDs) with masking + Drain."""
    cfg = load_config(config)
    typer.echo(f"[{cfg['dataset']}] Parsing {cfg['paths']['artifacts_dir']}/logs.parquet ...")
    stats = parse_dataset(cfg)
    typer.echo(f"  lines:                  {stats.lines:,}")
    typer.echo(f"  distinct messages:      {stats.distinct_messages:,}")
    typer.echo(f"  distinct after masking: {stats.distinct_masked:,}  (what Drain had to read)")
    typer.echo(f"  templates found:        {stats.templates:,}")
    if stats.grouping_accuracy is not None:
        typer.echo(f"  ground-truth templates: {stats.true_templates:,}")
        typer.echo(f"  grouping accuracy:      {stats.grouping_accuracy:.2%}")
    typer.echo(f"Saved events, templates and parser in {cfg['paths']['artifacts_dir']}")
    typer.echo(f"Template table: {cfg['paths']['reports_dir']}/templates.md")


@app.command()
@friendly_errors
def features(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Build the event-count matrix (one row per block) with time-based splits."""
    cfg = load_config(config)
    typer.echo(f"[{cfg['dataset']}] Building features from {cfg['paths']['artifacts_dir']} ...")
    stats = build_features(cfg)
    typer.echo(f"  matrix: {stats.blocks:,} blocks x {stats.events} events")
    typer.echo(stats.split_summary.to_string())
    typer.echo(f"Saved {cfg['paths']['artifacts_dir']}/features.parquet")
    typer.echo(f"Summary: {cfg['paths']['reports_dir']}/features.md")


@app.command()
@friendly_errors
def train(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Train the anomaly detectors on the train split (unsupervised: no labels)."""
    cfg = load_config(config)
    typer.echo(
        f"[{cfg['dataset']}] Training on the train split of {cfg['paths']['artifacts_dir']} ..."
    )
    results = train_models(cfg)

    def fmt(value: float | None) -> str:
        return "  n/a" if value is None else f"{value:.3f}"

    typer.echo("  Quick check on the validation split (full evaluation: Stage 5)")
    typer.echo(f"  {'model':<18}{'fit (s)':>8}{'ROC-AUC':>9}{'PR-AUC':>8}")
    for r in results:
        typer.echo(
            f"  {r.model:<18}{r.fit_seconds:>8.1f}{fmt(r.val_roc_auc):>9}{fmt(r.val_pr_auc):>8}"
        )
    typer.echo("  ROC-AUC: 0.5 = random guessing, 1.0 = perfect ranking.")
    if results and results[0].val_anomaly_share is not None:
        share = results[0].val_anomaly_share
        typer.echo(f"  PR-AUC: random guessing = share of anomalies = {share:.3f}, 1.0 = perfect.")
    typer.echo(f"Saved models and scores.parquet in {cfg['paths']['artifacts_dir']}")


@app.command()
@friendly_errors
def evaluate(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Pick thresholds on validation, then measure every model ONCE on the test split."""
    cfg = load_config(config)
    typer.echo(f"[{cfg['dataset']}] Evaluating {cfg['paths']['artifacts_dir']}/scores.parquet ...")
    results = evaluate_scores(cfg)
    typer.echo("  Test split (thresholds chosen on validation; 95% bootstrap intervals)")
    typer.echo(
        f"  {'model':<18}{'precision':>10}{'recall':>8}{'F1':>7}  {'F1 95% CI':<13}"
        f"{'caught':>7}{'false alarms':>13}{'missed':>7}"
    )
    for r in results.itertuples():
        ci = f"{r.f1_ci[0]:.3f}-{r.f1_ci[1]:.3f}"
        typer.echo(
            f"  {r.model:<18}{r.precision:>10.3f}{r.recall:>8.3f}{r.f1:>7.3f}  {ci:<13}"
            f"{r.tp:>7,}{r.fp:>13,}{r.fn:>7,}"
        )
    typer.echo(
        f"Report: {cfg['paths']['reports_dir']}/results.md (+ pr_curves.png, "
        "score_distributions.png)"
    )


@app.command("export-app")
@friendly_errors
def export_app(
    config: ConfigPath = DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option(help="Folder of the dashboard.")] = Path("app"),
) -> None:
    """Write the dashboard's model bundle (app/model) and demo log (app/demo)."""
    cfg = load_config(config)
    bundle_dir = export_bundle(cfg, out / "model")
    typer.echo(f"Model bundle: {bundle_dir}")
    log_path, labels_path = export_demo(cfg, out / "demo")
    typer.echo(
        f"Demo log: {log_path} ({log_path.stat().st_size / 1e6:.2f} MB), labels: {labels_path}"
    )


@app.command("run-all")
@friendly_errors
def run_all(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Run every step in order: download → ingest → parse → features → train → evaluate."""
    for step in (download, ingest, parse, features, train, evaluate):
        typer.echo()
        typer.secho(f"=== {step.__name__} ===", bold=True)
        step(config)


if __name__ == "__main__":
    app()
