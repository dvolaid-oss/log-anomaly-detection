import json

import numpy as np
import pandas as pd

from logad.evaluation.visuals import make_charts


def make_dataset(tmp_path, name: str, with_attacks: bool) -> dict:
    """A tiny dataset with everything `logad charts` reads: features, scores, thresholds,
    templates, and (optionally) per-line attack labels."""
    rng = np.random.default_rng(0)
    n = 400
    label = (rng.random(n) < 0.15).astype(int)
    counts = rng.poisson(3, size=(n, 3))
    counts[:, 2] = label * (5 + rng.poisson(3, size=n))  # E3: bursts ONLY in anomalies
    groups = [f"g{i}" for i in range(n)]
    features = pd.DataFrame(counts, columns=["E1", "E2", "E3"])
    features.insert(0, "group_id", groups)
    features.insert(1, "first_seen", pd.date_range("2026-01-01", periods=n, freq="min"))
    features.insert(2, "split", ["train"] * 240 + ["validation"] * 80 + ["test"] * 80)
    features.insert(3, "label", pd.array(label, dtype="Int8"))

    artifacts, raw = tmp_path / name / "artifacts", tmp_path / name / "raw"
    artifacts.mkdir(parents=True)
    raw.mkdir(parents=True)
    features.to_parquet(artifacts / "features.parquet", index=False)
    scores = features[["group_id", "split", "label"]].assign(
        score_pca=np.exp(rng.normal(0, 1, n) + 3 * label), score_baseline=rng.random(n)
    )
    scores.to_parquet(artifacts / "scores.parquet", index=False)
    thresholds = {
        "pca": {"threshold": 5.0, "validation_f1": 0.8},
        "baseline": {"threshold": 0.5, "validation_f1": 0.3},
    }
    (artifacts / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    pd.DataFrame({"event_id": ["E1", "E2", "E3"], "template": ["a", "b", "burst"]}).to_csv(
        artifacts / "templates.csv", index=False
    )

    config = {
        "dataset": name,
        "paths": {"artifacts_dir": str(artifacts), "raw_dir": str(raw)},
        "grouping": {"method": "session"},
        "data": {},
    }
    if with_attacks:  # one line per group, attack type on anomalous groups
        pd.DataFrame(
            {
                "line_id": range(n),
                "timestamp": features["first_seen"],
                "group_id": groups,
                "event_id": "E3",
                "label": label,
            }
        ).to_parquet(artifacts / "events.parquet", index=False)
        attack = np.where(label == 1, np.where(rng.random(n) < 0.5, "brute_force", "spray"), "")
        pd.DataFrame({"line_id": range(n), "attack": attack}).to_csv(
            raw / "labels.csv", index=False
        )
        config["data"]["label_file"] = "labels.csv"
    return config


def test_make_charts_writes_all_figures_and_facts(tmp_path):
    configs = [
        make_dataset(tmp_path, "hdfs", with_attacks=False),
        make_dataset(tmp_path, "auth", with_attacks=True),
    ]
    out = tmp_path / "visuals"
    facts = make_charts(configs, out)

    for name in ["pca_scatter", "box_plot", "distribution", "bubble_chart", "box_plot_attacks"]:
        assert (out / f"{name}.png").stat().st_size > 10_000, name
    saved = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    assert saved["summary"] == facts["summary"]

    # The best model on VALIDATION (pca, F1 0.8) is the one charted, not the baseline.
    assert facts["box_plot"]["hdfs"]["model"] == "pca"
    assert facts["summary"]["hdfs"]["test_groups"] == 80
    # E3 bursts only in anomalies -> it's the top anomaly signal in the bubble chart.
    assert facts["bubble"]["hdfs"]["top"][0]["event"] == "E3"
    assert set(facts["attack_box_plot"]) >= {"normal", "brute_force", "spray"}
