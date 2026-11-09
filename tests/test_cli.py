from pathlib import Path

from typer.testing import CliRunner

from logad import __version__
from logad.cli import app

runner = CliRunner()
HDFS_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "hdfs.yaml")


def test_help_lists_all_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["download", "ingest", "parse", "features", "train", "evaluate", "run-all"]:
        assert command in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_show_config_prints_dataset():
    result = runner.invoke(app, ["show-config", "--config", HDFS_CONFIG])
    assert result.exit_code == 0
    assert "dataset: hdfs" in result.output


def test_missing_input_gives_one_clear_error_line(tmp_path):
    config = tmp_path / "c.yaml"
    lines = ["dataset: hdfs", "seed: 1", "paths:", f"  artifacts_dir: {tmp_path.as_posix()}"]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--config", str(config)])
    assert result.exit_code == 1
    assert "logad train" in result.output
    assert "Traceback" not in result.output
