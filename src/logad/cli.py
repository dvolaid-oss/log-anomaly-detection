"""Command-line interface for the pipeline.

Usage:
    logad show-config --config configs/hdfs.yaml
    logad download --config configs/hdfs.yaml
    logad ingest --config configs/hdfs.yaml

Each pipeline stage is one command. Commands print a "not built yet" message until
their stage is implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from logad import __version__
from logad.adapters import ingest_dataset
from logad.config import load_config
from logad.download import download_dataset

app = typer.Typer(
    help="Log anomaly detection pipeline: download → ingest → parse → features → train → evaluate.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_CONFIG = Path("configs/hdfs.yaml")
ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config file.")]


def _not_built_yet(step: str, stage: int, config_path: Path) -> None:
    """Validate the config, then explain that this step is still a placeholder."""
    config = load_config(config_path)
    typer.echo(f"[{config['dataset']}] '{step}' is not built yet (coming in Stage {stage}).")
    raise typer.Exit(code=1)


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
def show_config(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Print the loaded config, to check which settings a run will use."""
    cfg = load_config(config)
    typer.echo(f"dataset: {cfg['dataset']}")
    typer.echo(f"seed:    {cfg['seed']}")
    for name, value in cfg["paths"].items():
        typer.echo(f"paths.{name}: {value}")


@app.command()
def download(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Download the raw dataset from official LogHub sources and verify checksums."""
    cfg = load_config(config)
    download_dataset(cfg, report=typer.echo)
    typer.echo(f"[{cfg['dataset']}] Raw data is ready in {cfg['paths']['raw_dir']}")


@app.command()
def ingest(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Convert raw logs to the common schema and save them as Parquet."""
    cfg = load_config(config)
    typer.echo(
        f"[{cfg['dataset']}] Ingesting {cfg['paths']['raw_dir']} (may take a few minutes)..."
    )
    out_path, stats = ingest_dataset(cfg)
    typer.echo(f"  lines read:           {stats.lines:,}")
    typer.echo(f"  lines skipped (bad):  {stats.skipped_lines:,}")
    typer.echo(f"  rows written:         {stats.rows:,}")
    typer.echo(f"  lines without block:  {stats.lines_without_group:,}")
    typer.echo(f"  groups (blocks):      {stats.groups:,}")
    typer.echo(f"  anomalous groups:     {stats.anomalous_groups:,}")
    typer.echo(f"  unlabelled groups:    {stats.unlabelled_groups:,}")
    typer.echo(f"Saved {out_path}")


@app.command()
def parse(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Parse raw logs into templates and event IDs with Drain."""
    _not_built_yet("parse", 2, config)


@app.command()
def features(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Build feature matrices (event counts per block/window, TF-IDF)."""
    _not_built_yet("features", 3, config)


@app.command()
def train(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Train the anomaly detectors (unsupervised: no labels)."""
    _not_built_yet("train", 4, config)


@app.command()
def evaluate(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Evaluate detectors against the labels and write reports."""
    _not_built_yet("evaluate", 5, config)


@app.command("run-all")
def run_all(config: ConfigPath = DEFAULT_CONFIG) -> None:
    """Run every step in order: download → ingest → parse → features → train → evaluate."""
    _not_built_yet("run-all", 5, config)


if __name__ == "__main__":
    app()
