import pickle

import numpy as np
import pandas as pd
import pytest

from logad.features import (
    assign_splits,
    build_features,
    event_columns,
    load_features,
    transformer_from_config,
)
from logad.features.transform import TfidfWeighting, make_transformer


def test_event_columns_sorted_numerically():
    assert event_columns(["E10", "label", "E2", "E1", "Extra"]) == ["E1", "E2", "E10"]


def test_assign_splits_in_order():
    splits = assign_splits(10, train=0.6, validation=0.2)
    assert list(splits) == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2


@pytest.mark.parametrize(("train", "validation"), [(0, 0.2), (1.0, 0), (0.7, 0.3), (0.6, -0.1)])
def test_assign_splits_rejects_bad_shares(train, validation):
    with pytest.raises(ValueError):
        assign_splits(10, train, validation)


def make_events(tmp_path):
    """A tiny events.parquet: 4 blocks, deliberately not in time order in the file."""
    ts = pd.Timestamp("2008-11-09 20:00:00")
    rows = [
        # line_id, minutes, group_id, event_id, label
        (0, 30, "blk_c", "E1", 0),
        (1, 30, "blk_c", "E2", 0),
        (2, 0, "blk_a", "E1", 0),
        (3, 1, "blk_a", "E1", 0),
        (4, 2, "blk_a", "E10", 0),
        (5, 10, "blk_b", "E2", 1),
        (6, 40, "blk_d", "E1", 0),
        (7, 50, None, "E2", None),  # a line without a block is ignored
    ]
    events = pd.DataFrame(rows, columns=["line_id", "minutes", "group_id", "event_id", "label"])
    events["timestamp"] = ts + pd.to_timedelta(events.pop("minutes"), unit="min")
    events["label"] = events["label"].astype("Int8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    events.to_parquet(artifacts / "events.parquet", index=False)
    return {
        "dataset": "hdfs",
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "artifacts_dir": str(artifacts),
            "reports_dir": str(tmp_path / "reports"),
        },
        "features": {"split": {"train": 0.5, "validation": 0.25}},
    }


def test_build_features_counts_events_per_block_in_time_order(tmp_path):
    config = make_events(tmp_path)
    stats = build_features(config)
    df, events = load_features(config)

    assert events == ["E1", "E2", "E10"]
    assert list(df["group_id"]) == ["blk_a", "blk_b", "blk_c", "blk_d"]  # sorted by first_seen
    assert df.loc[0, events].tolist() == [2, 0, 1]  # blk_a: E1 twice, E10 once
    assert list(df["split"]) == ["train", "train", "validation", "test"]
    assert df.loc[df["group_id"] == "blk_b", "label"].item() == 1
    assert (stats.blocks, stats.events) == (4, 3)
    assert stats.split_summary.loc["train", "anomalies"] == 1
    report = (tmp_path / "reports" / "features.md").read_text(encoding="utf-8")
    assert "4 blocks × 3 events" in report


def test_build_features_without_parse_explains_what_to_run(tmp_path):
    config = {"paths": {"artifacts_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="logad parse"):
        build_features(config)


def test_tfidf_weights_rare_events_higher():
    # Event 0 appears in every block, event 1 in one block only.
    X = np.array([[1, 0], [2, 0], [1, 3]])
    weighting = TfidfWeighting().fit(X)
    assert weighting.idf_[0] == pytest.approx(1.0)  # ln(4/4) + 1
    assert weighting.idf_[1] > weighting.idf_[0]
    assert weighting.transform(X)[2, 1] == pytest.approx(3 * weighting.idf_[1])


def test_transformer_learns_only_from_fit_data():
    train = np.array([[1.0, 0.0], [3.0, 0.0]])
    test = np.array([[100.0, 5.0]])
    transformer = make_transformer("event_count", scale=True).fit(train)
    # Mean/spread come from train only: column 0 mean 2, std 1 → (100 - 2) / 1.
    assert transformer.transform(test)[0, 0] == pytest.approx(98.0)


def test_plain_counts_pass_through_as_floats():
    transformer = make_transformer("event_count", scale=False).fit([[1, 2]])
    out = transformer.transform([[3, 4]])
    assert out.dtype == float
    assert out.tolist() == [[3.0, 4.0]]


def test_transformer_from_config_and_bad_type():
    pipeline = transformer_from_config({"features": {"type": "tfidf", "scale": True}})
    assert list(pipeline.named_steps) == ["tfidf", "scale"]
    with pytest.raises(ValueError, match="features.type"):
        make_transformer("word2vec")


def test_fitted_transformer_can_be_saved():
    # Stage 4 saves fitted models + transformers to disk, so they must be picklable.
    for feature_type in ["event_count", "tfidf"]:
        fitted = make_transformer(feature_type).fit([[1, 2], [0, 1]])
        restored = pickle.loads(pickle.dumps(fitted))
        assert restored.transform([[1, 1]]).tolist() == fitted.transform([[1, 1]]).tolist()
