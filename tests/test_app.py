"""Explanations, inference on raw logs, and the dashboard itself.

Uses the committed model bundle (app/model) and demo log (app/demo), so these tests also
run in CI, where the full dataset isn't available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logad.explain import Explainer, short
from logad.inference import (
    analyze_lines,
    block_lines,
    expected_vs_found,
    explain_block,
    load_bundle,
)
from logad.models.pca import PCADetector

ROOT = Path(__file__).resolve().parents[1]
DEMO_LOG = ROOT / "app" / "demo" / "hdfs_demo.log"
DEMO_LABELS = ROOT / "app" / "demo" / "hdfs_demo_labels.csv"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT / "app" / "model")


@pytest.fixture(scope="module")
def demo(bundle):
    return analyze_lines(DEMO_LOG.read_text(encoding="utf-8").splitlines(), bundle)


# ---------------------------------------------------------------- explanations (synthetic)
def make_explainer():
    rng = np.random.default_rng(0)
    k = rng.integers(1, 4, size=300)
    X = np.column_stack([3 * k, 3 * k, 3 * k, np.zeros(300)]).astype(float)
    pca = PCADetector(transform="event_count").fit(X)
    share = (X > 0).mean(axis=0)
    templates = {"E1": "receive", "E2": "respond", "E3": "store", "E4": "error"}
    return Explainer(pca, ["E1", "E2", "E3", "E4"], templates, share)


def test_rare_event_reason():
    reasons = make_explainer().explain([6, 6, 6, 1])
    assert reasons[0].kind == "rare"
    assert "E4 'error'" in reasons[0].text and "never seen" in reasons[0].text


def test_missing_events_reason():
    reasons = make_explainer().explain([6, 0, 0, 0])  # receive happened, the rest didn't
    kinds = {r.kind for r in reasons}
    assert "missing" in kinds
    assert any("E2 'respond'" in r.text and "found 0" in r.text for r in reasons)


def test_rare_event_not_reported_twice():
    reasons = make_explainer().explain([6, 6, 6, 5])
    assert sum(r.event == "E4" for r in reasons) == 1


def test_unknown_lines_reason_and_short():
    reasons = make_explainer().explain([6, 6, 6, 0], unknown_lines=2)
    assert reasons[-1].kind == "unknown"
    assert short("x" * 100, limit=10) == "xxxxxxx..."


# ---------------------------------------------------------------- inference on the demo log
def test_bundle_matches_training_setup(bundle):
    assert len(bundle.events) == len(bundle.train_share) == 45
    assert bundle.info["model"] == "pca"
    assert bundle.threshold > 0


def test_demo_is_scored_like_the_test_split(demo):
    labels = pd.read_csv(DEMO_LABELS).set_index("BlockId")["Label"] == "Anomaly"
    blocks = demo.blocks.set_index("group_id").join(labels.rename("anomaly"))
    assert len(blocks) == 300 and demo.skipped_lines == 0
    caught = int((blocks["alert"] & blocks["anomaly"]).sum())
    false_alarms = int((blocks["alert"] & ~blocks["anomaly"]).sum())
    assert caught >= 25  # 27 of 30 when the bundle was exported
    assert false_alarms <= 2  # 0 when the bundle was exported


def test_flagged_blocks_have_reasons(demo, bundle):
    for group_id in demo.blocks.loc[demo.blocks["alert"], "group_id"]:
        assert explain_block(demo, bundle, group_id), group_id


def test_short_dead_block_is_explained_by_missing_events(bundle):
    lines = [
        "081111 091000 1 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_42 src: /10.0.0.1:1 dest: /10.0.0.1:50010",
        "081111 091000 2 INFO dfs.FSNamesystem: "
        "BLOCK* NameSystem.allocateBlock: /user/root/x/part-1. blk_42",
    ]
    analysis = analyze_lines(lines, bundle)
    assert bool(analysis.blocks.loc[0, "alert"])
    reasons = explain_block(analysis, bundle, "blk_42")
    assert reasons[0].kind == "missing"


def test_detail_tables(demo, bundle):
    group_id = demo.blocks.iloc[0]["group_id"]
    table = expected_vs_found(demo, bundle, group_id)
    assert {"event", "found", "expected", "difference"} <= set(table.columns)
    lines = block_lines(demo, group_id)
    assert len(lines) == demo.blocks.iloc[0]["n_lines"]


def test_unknown_message_is_counted_not_crashing(bundle):
    lines = ["081111 091000 1 INFO dfs.DataNode: Something totally new happened to blk_7 today"]
    analysis = analyze_lines(lines, bundle)
    assert analysis.blocks.loc[0, "unknown_lines"] == 1


def test_non_hdfs_input_gives_clear_error(bundle):
    with pytest.raises(ValueError, match="only understands HDFS"):
        analyze_lines(["Nov 11 10:00:00 host sshd[1]: Failed password for root"], bundle)


# ---------------------------------------------------------------- the dashboard itself
def test_dashboard_demo_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Blocks"] == "300"
    assert metrics["Caught"].endswith("/30")
