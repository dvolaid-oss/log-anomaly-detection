import ipaddress
import re
from pathlib import Path

import pandas as pd
import pytest

from logad.adapters import ingest_dataset
from logad.adapters.auth_log import parse_lines
from logad.download import download_dataset
from logad.evaluation import evaluate_scores
from logad.features import build_features, group_expression
from logad.models import train_models
from logad.parsing import parse_dataset
from logad.synth.auth_log import ATTACKS

DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]


def make_config(tmp_path, days=20, attacks_per_type=8):
    return {
        "dataset": "auth",
        "seed": 1,
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": {
            "generate": {
                "generator": "auth_log",
                "seed": 1,
                "start": "2026-03-02",
                "days": days,
                "scanners_per_day": 20,
                "attacks_per_type": attacks_per_type,
            },
            "log_file": "auth.log",
            "label_file": "auth_labels.csv",
            "adapter_options": {"year": 2026},
        },
        "grouping": {"method": "session_window", "window_minutes": 10},
        "parsing": {
            "masks": [
                {"name": "IP", "pattern": r"(\d+\.){3}\d+"},
                {"name": "NUM", "pattern": r"\d+"},
            ]
        },
        "features": {"split": {"train": 0.6, "validation": 0.2}},
        "models": {"baseline": {}},
    }


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    config = make_config(tmp_path_factory.mktemp("auth"))
    download_dataset(config, report=lambda msg: None)  # runs the generator
    return config


def raw_files(config):
    raw = Path(config["paths"]["raw_dir"])
    log = (raw / "auth.log").read_text(encoding="utf-8").splitlines()
    labels = pd.read_csv(raw / "auth_labels.csv", keep_default_na=False)
    return log, labels


def test_generator_is_reproducible(tmp_path, generated):
    again = make_config(tmp_path)
    download_dataset(again, report=lambda msg: None)
    assert raw_files(again)[0] == raw_files(generated)[0]


def test_generator_labels_every_line_and_injects_every_attack(generated):
    log, labels = raw_files(generated)
    assert len(log) == len(labels)
    assert set(ATTACKS) <= set(labels["attack"])
    assert (labels["attack"] == "").mean() > 0.5  # most traffic is normal


def test_generator_uses_only_safe_ip_addresses(generated):
    log, _ = raw_files(generated)
    for ip in set(re.findall(r"\b\d+\.\d+\.\d+\.\d+\b", "\n".join(log))):
        address = ipaddress.ip_address(ip)
        assert address.is_private or any(address in net for net in DOC_NETS), ip


def test_parse_auth_lines():
    lines = [
        "Mar  2 14:02:31 web01 sshd[2345]: Failed password for root from 203.0.113.7 port 42 ssh2",
        "Mar 12 09:00:00 web01 sshd[99]: pam_unix(sshd:session): session closed for user bo",
        "this is not syslog",
    ]
    df, skipped = parse_lines(lines, year=2026)
    assert skipped == 1
    assert df.loc[0, "timestamp"] == pd.Timestamp("2026-03-02 14:02:31")
    assert df.loc[0, "group_id"] == "203.0.113.7"
    assert df.loc[0, "component"] == "sshd"
    assert df.loc[0, "message"].startswith("Failed password for root")
    assert pd.isna(df.loc[1, "group_id"])  # no IP in the line → not grouped
    assert df.loc[1, "timestamp"] == pd.Timestamp("2026-03-12 09:00:00")


def test_session_window_groups_by_ip_and_window():
    expression = group_expression({"grouping": {"method": "session_window", "window_minutes": 10}})
    assert expression.startswith("group_id || '@' ||") and "10 MINUTE" in expression


def test_full_pipeline_with_attack_breakdown(generated):
    _, stats = ingest_dataset(generated)
    assert stats.attack_lines > 0 and stats.source_ips > 10
    parse_dataset(generated)
    features = build_features(generated)
    assert features.blocks > 100
    train_models(generated)
    results = evaluate_scores(generated, n_resamples=100)

    breakdown = results.attrs["breakdown"]
    assert set(breakdown["attack"]) <= set(ATTACKS)
    assert (breakdown["baseline"] <= breakdown["groups"]).all()
    report = (Path(generated["paths"]["reports_dir"]) / "results.md").read_text(encoding="utf-8")
    assert "Caught per attack type" in report
