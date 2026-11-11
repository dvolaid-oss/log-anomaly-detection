import pandas as pd
import pytest

from logad.adapters import ingest_dataset
from logad.adapters.base import COLUMNS
from logad.adapters.bgl import parse_lines
from logad.config import load_config
from logad.features import build_features, group_expression, load_features
from logad.parsing.drain import LogParser

NORMAL = (
    "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
    "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected\n"
)
ALERT = (
    "KERNDTLB 1117869872 2005.06.04 R23-M0-NE-C:J05-U01 2005-06-04-00.24.32.432192 "
    "R23-M0-NE-C:J05-U01 RAS KERNEL FATAL data TLB error interrupt\n"
)
NO_MESSAGE = "- 1117838600 2005.06.03 NULL 2005-06-03-15.43.20.000000 NULL RAS KERNEL INFO\n"


def test_parse_fields():
    df, skipped = parse_lines([NORMAL])
    assert skipped == 0 and list(df.columns) == COLUMNS
    row = df.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2005-06-03 22:42:50")  # unix time is UTC
    assert (row["level"], row["component"], row["source"]) == ("INFO", "KERNEL", "bgl")
    assert row["message"] == "instruction cache parity error corrected"
    assert pd.isna(row["group_id"])  # windows come later
    assert row["label"] == 0


def test_label_never_leaks_into_the_message():
    # The alert category is the first word of the raw line: it must end up ONLY in `label`.
    df, _ = parse_lines([ALERT])
    assert df.loc[0, "label"] == 1
    assert "KERNDTLB" not in df.loc[0, "message"]
    assert df.loc[0, "message"] == "data TLB error interrupt"


def test_line_without_message_and_garbage():
    df, skipped = parse_lines([NO_MESSAGE, "not a BGL line\n"])
    assert skipped == 1
    assert df.loc[0, "message"] == ""


def test_ingest_dataset_uses_bgl_adapter(tmp_path):
    (tmp_path / "BGL.log").write_text(NORMAL + ALERT + NORMAL, encoding="utf-8")
    config = {
        "dataset": "bgl",
        "paths": {"raw_dir": str(tmp_path), "artifacts_dir": str(tmp_path / "a")},
        "data": {"log_file": "BGL.log", "label_file": None},
    }
    out, stats = ingest_dataset(config)
    assert (stats.lines, stats.anomalous_lines) == (3, 1)
    assert len(pd.read_parquet(out)) == 3


def test_bgl_masks_from_config():
    parser = LogParser.from_config(load_config("configs/bgl.yaml"))
    assert parser.mask("CE sym 2, at 0x0b85eee0, mask 0x05") == "CE sym <NUM>, at <HEX>, mask <HEX>"
    # bare 8-char values get ONE placeholder, decimal or hex (fixed BGL parsing to 96.85%)
    assert parser.mask("23431872 double-hummer") == parser.mask("63543 double-hummer")
    assert parser.mask("iar 00106570 dear 0247c8b4") == "iar <NUM> dear <NUM>"
    assert parser.mask("socket to 172.16.96.116:33569") == "socket to <IP>"
    assert parser.mask("on node R02-M1-N0-C:J12-U11") == "on node <NODE>"


def test_group_expression_options():
    assert group_expression({"grouping": {"method": "session"}}) == "group_id"
    assert "60 MINUTE" in group_expression(
        {"grouping": {"method": "time_window", "window_minutes": 60}}
    )
    with pytest.raises(ValueError):
        group_expression({"grouping": {"method": "sliding"}})
    with pytest.raises(ValueError):
        group_expression({"grouping": {"method": "time_window", "window_minutes": 0}})


def test_time_windows_count_events_and_label_any_alert(tmp_path):
    t0 = pd.Timestamp("2005-06-03 15:00:00")
    rows = [  # line_id, minutes after t0, event_id, label
        (0, 1, "E1", 0),
        (1, 20, "E1", 0),
        (2, 59, "E2", 1),  # still in the 15:00 window → that window is anomalous
        (3, 61, "E1", 0),  # 16:00 window
        (4, 130, "E2", 0),  # 17:00 window
        (5, 179, "E1", 0),
        (6, 200, "E1", 0),  # 18:00 window
        (7, 230, "E2", 0),
    ]
    events = pd.DataFrame(rows, columns=["line_id", "minutes", "event_id", "label"])
    events["timestamp"] = t0 + pd.to_timedelta(events.pop("minutes"), unit="min")
    events["group_id"] = None
    events["label"] = events["label"].astype("int8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    events.to_parquet(artifacts / "events.parquet", index=False)
    config = {
        "dataset": "bgl",
        "paths": {
            "raw_dir": str(tmp_path),
            "artifacts_dir": str(artifacts),
            "reports_dir": str(tmp_path / "reports"),
        },
        "grouping": {"method": "time_window", "window_minutes": 60},
        "features": {"split": {"train": 0.5, "validation": 0.25}},
    }

    stats = build_features(config)
    df, ev = load_features(config)

    assert list(df["group_id"]) == [
        "2005-06-03 15:00",
        "2005-06-03 16:00",
        "2005-06-03 17:00",
        "2005-06-03 18:00",
    ]
    assert df.loc[0, ev].tolist() == [2, 1]  # 15:00 window: E1 twice, E2 once
    assert df["label"].tolist() == [1, 0, 0, 0]
    assert list(df["split"]) == ["train", "train", "validation", "test"]
    # regression: 100 x anomalies used to overflow the 8-bit label type (negative %)
    assert stats.split_summary.loc["train", "anomaly_pct"] == 50.0
    report = (tmp_path / "reports" / "features.md").read_text(encoding="utf-8")
    assert "60-minute windows" in report
