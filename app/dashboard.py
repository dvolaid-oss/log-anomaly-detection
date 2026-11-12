"""Log Anomaly Detector: Streamlit dashboard with two tabs.

Run locally:   streamlit run app/dashboard.py

- HDFS (big-data logs): the PCA model, best on validation, test F1 0.980.
- SSH attacks (security): synthetic auth.log with injected attacks, scored by the
  autoencoder (best on validation), run in plain NumPy so no PyTorch is needed online.

Each tab parses a log (demo or upload) with the saved Drain parser, groups the lines (HDFS
blocks / source-IP 10-minute windows), scores every group, flags those above the threshold
chosen on the validation split, and explains each flag in plain words.

Safety: uploads are read into memory and never written to disk; the upload size is capped
in .streamlit/config.toml; log text is only shown inside tables or escaped markdown.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _source_fingerprint() -> tuple:
    """Name, size and modification time of every logad source file."""
    return tuple(
        sorted(
            (str(p.relative_to(SRC)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in SRC.rglob("*.py")
        )  # noqa: E501
    )


# Always run THIS repo's logad code. Two ways a hosted app (Streamlit Cloud) can run stale code
# after a push: an older installed copy of the package shadowing src/, or old logad modules
# still in memory (Streamlit reruns this script in a long-lived process). Seen after the
# Stage 9 push: the new dashboard called the Stage 7 inference code.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_fingerprint = _source_fingerprint()
_loaded = sys.modules.get("logad")
# Reload unless logad was loaded by THIS dashboard from THESE files: an installed copy, modules
# loaded by an older dashboard version (no fingerprint: the exact Stage 9 case), or files
# changed since loading all trigger a fresh import.
if _loaded is not None and getattr(_loaded, "_source_fingerprint", None) != _fingerprint:
    for _name in [m for m in sys.modules if m == "logad" or m.startswith("logad.")]:
        del sys.modules[_name]
    st.cache_resource.clear()  # cached bundles hold objects from the old code
import logad  # noqa: E402

logad._source_fingerprint = _fingerprint

from logad.inference import (  # noqa: E402
    analyze_lines,
    block_lines,
    expected_vs_found,
    explain_block,
    load_bundle,
)

REPO_URL = "https://github.com/dvolaid-oss/log-anomaly-detection"
FLAGGED, NOT_FLAGGED = "#eb6834", "#2a78d6"  # validated palette: orange, blue
KIND_ICON = {"rare": "🔸", "high": "📈", "missing": "🕳️", "extra": "➕", "unknown": "❔"}
MODEL_NAMES = {
    "pca": "PCA (Xu et al. 2009)",
    "autoencoder": "Autoencoder (neural network, run in NumPy)",
    "baseline": "Rule baseline",
    "isolation_forest": "Isolation Forest",
}
DATASETS = {
    "hdfs": {
        "tab": "📦 HDFS (big-data logs)",
        "demo": "Demo: 300 HDFS blocks",
        "demo_note": "Public LogHub data from blocks the model never trained on (test split).",
        "test_note": "test F1 0.980 on 115,012 unseen blocks",
        "intro": "Finds unusual sessions in Hadoop (HDFS) logs **without ever being shown "
        "labelled anomalies**, and explains each alert: which rare events a block contains, "
        "and which expected events are missing.",
    },
    "auth": {
        "tab": "🔐 SSH attacks (security)",
        "demo": "Demo: 2 days of SSH logs (synthetic)",
        "demo_note": "Synthetic auth.log from the test period (never trained on), with "
        "injected attacks among normal logins and background bot scanning.",
        "test_note": "test F1 0.728 on 2,961 unseen IP windows",
        "intro": "Finds attacks in SSH login logs (`auth.log`): brute force, password "
        "spraying, break-ins. Lines are grouped per **source IP and 10-minute window**. "
        "Background bot scanning counts as normal noise, as in a real SOC.",
    },
}


def title(text: str) -> str:
    """'IP window' -> 'IP window', 'block' -> 'Block' (str.capitalize would give 'Ip window')."""
    return text[:1].upper() + text[1:]


def escape_markdown(text: str) -> str:
    """Show log text literally: escape characters markdown would treat as formatting or
    HTML (e.g. 'BLOCK*' would start italics, '<BLK>' would look like a tag)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!<>|~])", r"\\\1", text)


st.set_page_config(page_title="Log Anomaly Detector", page_icon="🔎", layout="wide")


@st.cache_resource
def get_bundle(dataset: str):
    return load_bundle(ROOT / "app" / "model" / dataset)


@st.cache_data(max_entries=6, show_spinner="Parsing and scoring the log ...")
def analyze(dataset: str, raw: bytes):
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return analyze_lines(lines, get_bundle(dataset))


def score_chart(blocks: pd.DataFrame, threshold: float, unit: str) -> alt.LayerChart:
    """Histogram of scores (log scale), flagged vs not flagged, with the threshold line."""
    floor = max(float(blocks["score"].min()), threshold / 1e4, 1e-9)
    data = blocks.assign(
        log_score=np.log10(blocks["score"].clip(lower=floor)),
        status=np.where(blocks["alert"], "flagged", "not flagged"),
    )
    color = alt.Color(
        "status:N",
        scale=alt.Scale(domain=["not flagged", "flagged"], range=[NOT_FLAGGED, FLAGGED]),
        legend=alt.Legend(title=None, orient="top"),
    )
    axis = alt.Axis(labelExpr="format(pow(10, datum.value), '.2~g')", title="anomaly score (log)")
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("log_score:Q", bin=alt.Bin(maxbins=40), axis=axis),
            y=alt.Y("count():Q", title=f"{unit}s", scale=alt.Scale(type="symlog")),
            color=color,
            tooltip=[alt.Tooltip("status:N"), alt.Tooltip("count():Q", title=f"{unit}s")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [np.log10(threshold)]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="threshold (log10)", format=".2f")])
    )
    return (bars + rule).properties(height=220)


def read_input(dataset: str, meta: dict):
    """Demo or upload, chosen inside the tab. Returns (raw bytes or None, labels or None)."""
    source = st.radio(
        "Log source", [meta["demo"], "Upload a log file"], horizontal=True, key=f"src_{dataset}"
    )
    if source == meta["demo"]:
        demo = ROOT / "app" / "demo"
        labels = pd.read_csv(demo / f"{dataset}_demo_labels.csv", keep_default_na=False)
        st.caption(meta["demo_note"])
        return (demo / f"{dataset}_demo.log").read_bytes(), labels.set_index("group_id")
    upload = st.file_uploader(
        "Log file (max 20 MB, processed in memory, never stored)",
        type=["log", "txt"],
        key=f"upload_{dataset}",
    )
    if upload is None:
        st.info("Choose a file, or switch back to the demo.")
        return None, None
    return upload.getvalue(), None  # in memory only, never saved


def render(dataset: str) -> None:
    meta, bundle = DATASETS[dataset], get_bundle(dataset)
    unit, info = bundle.group_name, bundle.info
    st.write(meta["intro"])
    st.caption(
        f"Model: **{MODEL_NAMES.get(info['model'], info['model'])}**, trained on "
        f"{info['train_groups']:,} {unit}s without labels · alert threshold "
        f"{bundle.threshold:.3g} (best F1 on validation: {info['validation_f1']:.3f}) · "
        f"{meta['test_note']}"
    )

    raw, labels = read_input(dataset, meta)
    if raw is None:
        return
    try:
        analysis = analyze(dataset, raw)
    except ValueError as error:
        st.error(str(error))
        return

    blocks = analysis.blocks.copy()
    if labels is not None:
        blocks["true label"] = blocks["group_id"].map(labels["Label"]).fillna("?")
        if "attack" in labels.columns:
            blocks["attack"] = blocks["group_id"].map(labels["attack"]).fillna("")

    cols = st.columns(5 if labels is not None else 3)
    cols[0].metric("Log lines", f"{analysis.total_lines:,}")
    cols[1].metric(f"{title(unit)}s", f"{len(blocks):,}")
    cols[2].metric("Flagged", f"{int(blocks['alert'].sum()):,}")
    if labels is not None:
        is_anomaly = blocks["true label"] == "Anomaly"
        caught = int((blocks["alert"] & is_anomaly).sum())
        cols[3].metric("Caught", f"{caught}/{int(is_anomaly.sum())}", help="true anomalies flagged")
        cols[4].metric("False alarms", f"{int((blocks['alert'] & ~is_anomaly).sum())}")
    if analysis.skipped_lines:
        st.warning(f"{analysis.skipped_lines:,} lines weren't recognised and were skipped.")
    if "attack" in blocks.columns:
        per_attack = blocks[blocks["attack"] != ""].groupby("attack")["alert"].agg(["sum", "size"])
        st.markdown(
            "**Caught per attack type:** "
            + " · ".join(f"{a}: {int(r['sum'])}/{int(r['size'])}" for a, r in per_attack.iterrows())
        )

    st.subheader("Score distribution")
    st.altair_chart(score_chart(blocks, bundle.threshold, unit), width="stretch")
    st.caption(f"{title(unit)}s right of the dashed line are flagged. Hover for counts.")

    st.subheader(f"{title(unit)}s, most suspicious first")
    explanations = {
        g: explain_block(analysis, bundle, g) for g in blocks.loc[blocks["alert"], "group_id"]
    }
    blocks["top reason"] = blocks["group_id"].map(
        lambda g: explanations[g][0].text if explanations.get(g) else ""
    )
    show_all = st.toggle(f"Show all {unit}s (not only flagged ones)", key=f"all_{dataset}")
    table = blocks if show_all else blocks[blocks["alert"]]
    columns = ["group_id", "score", "n_lines"]
    columns += [c for c in ("true label", "attack") if c in blocks.columns] + ["top reason"]
    selection = st.dataframe(
        table[columns],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"table_{dataset}",
        column_config={
            "group_id": unit,
            "score": st.column_config.NumberColumn("score", format="%.3g"),
            "n_lines": "lines",
        },
    )
    if table.empty:
        st.success(f"No {unit} crossed the alert threshold.")
        return
    rows = selection.selection.rows if selection else []
    chosen = table.iloc[rows[0]]["group_id"] if rows else table.iloc[0]["group_id"]
    st.caption(f"Click a row to inspect another {unit}.")

    row = blocks.set_index("group_id").loc[chosen]
    st.subheader(f"{title(unit)} {chosen}")
    status = "🚩 flagged" if row["alert"] else "not flagged"
    truth = f" · true label: **{row['true label']}**" if labels is not None else ""
    if row.get("attack"):
        truth += f" ({escape_markdown(row['attack'])})"
    st.markdown(
        f"Score **{row['score']:.3g}** (threshold {bundle.threshold:.3g}) · {status}{truth}"
    )

    reasons = explanations.get(chosen) or explain_block(analysis, bundle, chosen)
    st.markdown("**Why**")
    if reasons:
        st.markdown("\n".join(f"- {KIND_ICON[r.kind]} {escape_markdown(r.text)}" for r in reasons))
    else:
        st.markdown(
            "- No single event stands out: the model reacts to the overall combination of "
            "events (see the counts below)."
        )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Event counts: found vs expected by the model**")
        st.dataframe(
            expected_vs_found(analysis, bundle, chosen),
            hide_index=True,
            width="stretch",
            column_config={"template": st.column_config.TextColumn(width="medium")},
        )
    with right:
        st.markdown(f"**The {unit}'s log lines**")
        st.dataframe(block_lines(analysis, chosen), hide_index=True, width="stretch")


st.title("🔎 Log Anomaly Detector")
st.markdown(
    f"Unsupervised anomaly detection on system and security logs, with explanations. "
    f"[How it works (GitHub)]({REPO_URL})"
)
# A link can open a tab first: ...streamlit.app/?tab=auth shows the SSH tab first.
order = list(DATASETS)
wanted = st.query_params.get("tab")
if wanted in DATASETS:
    order = [wanted] + [d for d in order if d != wanted]
for tab, dataset in zip(st.tabs([DATASETS[d]["tab"] for d in order]), order, strict=True):
    with tab:
        render(dataset)
