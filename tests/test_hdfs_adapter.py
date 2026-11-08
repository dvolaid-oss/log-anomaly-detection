import pandas as pd
import pytest

from logad.adapters import ingest_dataset
from logad.adapters.base import COLUMNS
from logad.adapters.hdfs import ingest, load_labels, parse_lines

# Real-looking HDFS lines (format copied from LogHub; block IDs shortened).
NORMAL = (
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
    "PacketResponder 1 for block blk_111 terminating\n"
)
NEGATIVE_ID = "081109 203807 222 WARN dfs.DataNode: Got exception serving blk_-222 to /10.0.0.1\n"
MULTI_BLOCK = (
    "081111 065254 19 INFO dfs.FSNamesystem: BLOCK* ask 10.0.0.2:50010 to delete  "
    "blk_111 blk_333 blk_111\n"
)
NO_BLOCK = "081109 210000 19 INFO dfs.FSNamesystem: Roll Edit Log\n"
MALFORMED = "this is not an HDFS line\n"


def test_parse_single_line_fields():
    df, skipped = parse_lines([NORMAL])
    assert skipped == 0
    assert list(df.columns) == COLUMNS
    row = df.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2008-11-09 20:36:15")
    assert row["level"] == "INFO"
    assert row["component"] == "dfs.DataNode$PacketResponder"
    assert row["message"] == "PacketResponder 1 for block blk_111 terminating"
    assert row["group_id"] == "blk_111"
    assert row["source"] == "hdfs"


def test_negative_block_id_keeps_minus_sign():
    df, _ = parse_lines([NEGATIVE_ID])
    assert df.loc[0, "group_id"] == "blk_-222"


def test_multi_block_line_gives_one_row_per_distinct_block():
    df, _ = parse_lines([MULTI_BLOCK], first_line_id=7)
    assert sorted(df["group_id"]) == ["blk_111", "blk_333"]  # blk_111 twice → once
    assert set(df["line_id"]) == {7}  # both rows point back to the same raw line


def test_line_without_block_kept_with_empty_group():
    df, _ = parse_lines([NO_BLOCK])
    assert len(df) == 1
    assert pd.isna(df.loc[0, "group_id"])


def test_malformed_line_is_skipped_and_counted():
    df, skipped = parse_lines([NORMAL, MALFORMED, NEGATIVE_ID])
    assert skipped == 1
    assert list(df["line_id"]) == [0, 2]  # line IDs still match the raw file


def test_load_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("BlockId,Label\nblk_111,Normal\nblk_-222,Anomaly\n", encoding="utf-8")
    assert load_labels(path) == {"blk_111": 0, "blk_-222": 1}


def test_load_labels_rejects_unknown_values(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("BlockId,Label\nblk_111,Maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown label"):
        load_labels(path)


def test_ingest_writes_parquet_with_labels_across_chunks(tmp_path):
    log = tmp_path / "HDFS.log"
    log.write_text(NORMAL + NEGATIVE_ID + MULTI_BLOCK + NO_BLOCK + MALFORMED, encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("BlockId,Label\nblk_111,Normal\nblk_-222,Anomaly\n", encoding="utf-8")
    out = tmp_path / "out" / "logs.parquet"

    stats = ingest(log, out, labels, chunk_size=2)  # tiny chunks → tests the chunk loop

    df = pd.read_parquet(out)
    assert list(df.columns) == COLUMNS
    assert list(df["line_id"].unique()) == [0, 1, 2, 3]
    assert stats.lines == 5
    assert stats.skipped_lines == 1
    assert stats.rows == len(df) == 5  # 1 + 1 + 2 (multi-block) + 1 (no block)
    assert stats.lines_without_group == 1
    assert stats.groups == 3
    assert stats.anomalous_groups == 1
    assert stats.unlabelled_groups == 1  # blk_333 is not in the labels file
    by_block = df.dropna(subset=["group_id"]).groupby("group_id")["label"].first()
    assert by_block["blk_-222"] == 1
    assert by_block["blk_111"] == 0
    assert pd.isna(by_block["blk_333"])
    assert not out.with_name("logs.parquet.part").exists()


def test_ingest_without_labels_leaves_label_empty(tmp_path):
    log = tmp_path / "HDFS.log"
    log.write_text(NORMAL, encoding="utf-8")
    out = tmp_path / "logs.parquet"
    ingest(log, out)
    assert pd.read_parquet(out)["label"].isna().all()


def test_ingest_dataset_explains_missing_raw_data(tmp_path):
    config = {
        "dataset": "hdfs",
        "paths": {"raw_dir": str(tmp_path), "artifacts_dir": str(tmp_path / "a")},
        "data": {"log_file": "HDFS.log", "label_file": None},
    }
    with pytest.raises(FileNotFoundError, match="logad download"):
        ingest_dataset(config)
