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
AUTH_LOG = ROOT / "app" / "demo" / "auth_demo.log"
AUTH_LABELS = ROOT / "app" / "demo" / "auth_demo_labels.csv"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT / "app" / "model" / "hdfs")


@pytest.fixture(scope="module")
def auth_bundle():
    return load_bundle(ROOT / "app" / "model" / "auth")


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
    labels = pd.read_csv(DEMO_LABELS).set_index("group_id")["Label"] == "Anomaly"
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
    with pytest.raises(ValueError, match="No usable HDFS log lines"):
        analyze_lines(["Nov 11 10:00:00 host sshd[1]: Failed password for root"], bundle)


# ---------------------------------------------------------------- the dashboard itself
def run_dashboard(setup: str = "") -> dict:
    """Run the dashboard with Streamlit's AppTest in a SEPARATE process and return its metrics.

    Separate, because the dashboard reloads logad modules it didn't load itself, which would
    swap the modules out from under the other tests in this process.
    """
    import json
    import subprocess
    import sys

    script = (
        "import json, sys, types\n"
        f"{setup}\n"
        "from streamlit.testing.v1 import AppTest\n"
        f"app = AppTest.from_file({str(ROOT / 'app' / 'dashboard.py')!r}, default_timeout=120)\n"
        "app.run()\n"
        "print(json.dumps({'exceptions': [str(e.value) for e in app.exception],\n"
        "                  'tabs': len(app.tabs),\n"
        "                  'metrics': [[m.label, m.value] for m in app.metric]}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_dashboard_demo_runs_without_errors():
    app = run_dashboard()
    assert app["exceptions"] == []
    assert app["tabs"] == 2
    metrics = dict(app["metrics"])  # both tabs render their demo
    assert metrics["Blocks"] == "300"
    assert metrics["IP windows"] == "411"
    caught = [value for label, value in app["metrics"] if label == "Caught"]
    assert caught[0].endswith("/30") and caught[1].endswith("/35")


def test_dashboard_uses_no_deprecated_streamlit_options():
    # Streamlit logs (not raises) deprecations, so the AppTest above can't see them.
    source = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "use_container_width" not in source  # replaced by width="stretch"


# ---------------------------------------------------------------- SSH bundle (autoencoder)
def test_auth_bundle_runs_the_autoencoder_without_pytorch(auth_bundle):
    from logad.models.numpy_autoencoder import NumpyAutoencoder

    assert auth_bundle.info["model"] == "autoencoder"  # best on validation
    assert isinstance(auth_bundle.detector, NumpyAutoencoder)
    assert auth_bundle.group_name == "IP window"


def test_auth_demo_catches_attacks_and_explains_bursts(auth_bundle):
    analysis = analyze_lines(AUTH_LOG.read_text(encoding="utf-8").splitlines(), auth_bundle)
    labels = pd.read_csv(AUTH_LABELS, keep_default_na=False).set_index("group_id")
    blocks = analysis.blocks.set_index("group_id").join(labels)
    anomalous = blocks["Label"] == "Anomaly"
    assert (blocks["alert"] & anomalous).sum() >= 25  # 31 of 35 at export time
    assert all("@" in g for g in blocks.index)  # groups are IP@window
    burst = blocks[(blocks["attack"] == "compromise") & blocks["alert"]].index[0]
    reasons = explain_block(analysis, auth_bundle, burst)
    assert reasons and reasons[0].kind == "high"
    assert "Failed password" in reasons[0].text


def test_high_count_reason_and_non_rebuilding_models():
    from logad.models.baseline import RareEventDetector

    X = np.array([[1, 0], [2, 0], [1, 0], [3, 1]], dtype=float)
    detector = RareEventDetector().fit(X)  # does not rebuild blocks → no mix reasons
    explainer = Explainer(detector, ["E1", "E2"], {"E1": "a", "E2": "b"}, [1.0, 0.25], [3, 1])
    reasons = explainer.explain([40, 0])
    assert [r.kind for r in reasons] == ["high"]
    assert "40x E1 'a'" in reasons[0].text and "at most 3" in reasons[0].text
    assert explainer.mix_reasons(np.array([40.0, 0.0])) == []


def test_best_model_uses_validation_f1():
    from logad.inference import best_model

    thresholds = {"pca": {"validation_f1": 0.9}, "baseline": {"validation_f1": 0.95}}
    assert best_model(thresholds) == "baseline"


def test_dashboard_ignores_a_stale_installed_logad():
    # An older installed copy of the package must not shadow this repo's src/.
    setup = (
        "old = types.ModuleType('logad'); old.__file__ = '/site-packages/logad/__init__.py'; "
        "sys.modules['logad'] = old"
    )
    assert run_dashboard(setup)["exceptions"] == []


def test_dashboard_reloads_logad_left_in_memory_by_an_older_version():
    # The exact Stage 9 live failure: logad loaded from src/ by an OLDER dashboard (so no
    # fingerprint), still looking for the old model file name. It must be reloaded.
    setup = (
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import logad.inference; "
        "logad.inference.BUNDLE_MODEL = 'pca.joblib'"  # what the Stage 7 code expected
    )
    app = run_dashboard(setup)
    assert app["exceptions"] == []
    assert dict(app["metrics"])["Blocks"] == "300"
