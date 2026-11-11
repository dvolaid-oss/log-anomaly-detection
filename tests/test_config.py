from pathlib import Path

import pytest

from logad.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("name", "dataset"),
    [("hdfs", "hdfs"), ("hdfs_sample", "hdfs"), ("bgl", "bgl"), ("bgl_sample", "bgl")],
)
def test_project_configs_load(name, dataset):
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config["dataset"] == dataset
    assert isinstance(config["seed"], int)
    assert "artifacts_dir" in config["paths"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_missing_required_keys_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("dataset: hdfs\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(path)


def test_non_mapping_config_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
