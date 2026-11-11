"""Log Anomaly Detector: Streamlit dashboard.

Run locally:   streamlit run app/dashboard.py

Upload an HDFS log (or use the demo), and the app parses it with the saved Drain parser,
scores every block with the saved PCA model, flags blocks above the threshold chosen on the
validation split, and explains each flag in plain words.

Safety: uploads are read into memory and never written to disk; the upload size is capped
in .streamlit/config.toml; log text is only shown inside tables (never rendered as HTML).
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
try:
    import logad  # noqa: F401  (installed with `pip install -e .`)
except ImportError:  # e.g. on Streamlit Community Cloud, which only installs requirements.txt
    sys.path.insert(0, str(ROOT / "src"))

from logad.inference import (  # noqa: E402
    analyze_lines,
    block_lines,
    expected_vs_found,
    explain_block,
    load_bundle,
)

MODEL_DIR = ROOT / "app" / "model"
DEMO_LOG = ROOT / "app" / "demo" / "hdfs_demo.log"
DEMO_LABELS = ROOT / "app" / "demo" / "hdfs_demo_labels.csv"
REPO_URL = "https://github.com/dvolaid-oss/log-anomaly-detection"
FLAGGED, NOT_FLAGGED = "#eb6834", "#2a78d6"  # validated palette: orange, blue
KIND_ICON = {"rare": "🔸", "missing": "🕳️", "extra": "➕", "unknown": "❔"}


def escape_markdown(text: str) -> str:
    """Show log text literally: escape characters markdown would treat as formatting or
    HTML (e.g. 'BLOCK*' would start italics, '<BLK>' would look like a tag)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!<>|~])", r"\\\1", text)


st.set_page_config(page_title="Log Anomaly Detector", page_icon="🔎", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_DIR)


@st.cache_data(max_entries=5, show_spinner="Parsing and scoring the log ...")
def analyze(raw: bytes):
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return analyze_lines(lines, get_bundle())


def score_chart(blocks: pd.DataFrame, threshold: float) -> alt.LayerChart:
    """Histogram of scores (log scale), flagged vs not flagged, with the threshold line."""
    data = blocks.assign(
        log_score=np.log10(blocks["score"].clip(lower=1e-3)),
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
            y=alt.Y("count():Q", title="blocks", scale=alt.Scale(type="symlog")),
            color=color,
            tooltip=[alt.Tooltip("status:N"), alt.Tooltip("count():Q", title="blocks")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [np.log10(threshold)]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="threshold (log10)", format=".2f")])
    )
    return (bars + rule).properties(height=220)


bundle = get_bundle()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Input")
    source = st.radio("Log source", ["Demo: 300 HDFS blocks", "Upload an HDFS log"])
    labels = None
    if source.startswith("Demo"):
        raw = DEMO_LOG.read_bytes()
        labels = pd.read_csv(DEMO_LABELS).set_index("BlockId")["Label"]
        st.caption("Public LogHub data from blocks the model never trained on (test split).")
    else:
        upload = st.file_uploader("HDFS log file (max 20 MB)", type=["log", "txt"])
        if upload is None:
            st.info("Choose a file, or switch back to the demo.")
            st.stop()
        raw = upload.getvalue()  # in memory only, never saved

    st.header("Model")
    info = bundle.info
    st.markdown(
        f"- **PCA** detector (Xu et al. 2009)\n"
        f"- trained on {info['train_blocks']:,} blocks, no labels\n"
        f"- alert threshold **{bundle.threshold:.3g}** (best F1 on validation: "
        f"{info['validation_f1']:.3f})\n"
        f"- test F1 0.980 on 115,012 unseen blocks"
    )
    st.markdown(f"[How it works (GitHub)]({REPO_URL})")

# ----------------------------------------------------------------------------- main
st.title("🔎 Log Anomaly Detector")
st.write(
    "Finds unusual sessions in HDFS logs **without ever being shown labelled anomalies**, and "
    "explains each alert: which rare events a block contains, and which expected events "
    "are missing."
)

try:
    analysis = analyze(raw)
except ValueError as error:
    st.error(str(error))
    st.stop()

blocks = analysis.blocks.copy()
if labels is not None:
    blocks["true label"] = blocks["group_id"].map(labels).fillna("?")

cols = st.columns(5 if labels is not None else 3)
cols[0].metric("Log lines", f"{analysis.total_lines:,}")
cols[1].metric("Blocks", f"{len(blocks):,}")
cols[2].metric("Flagged", f"{int(blocks['alert'].sum()):,}")
if labels is not None:
    is_anomaly = blocks["true label"] == "Anomaly"
    caught = int((blocks["alert"] & is_anomaly).sum())
    cols[3].metric("Caught", f"{caught}/{int(is_anomaly.sum())}", help="true anomalies flagged")
    cols[4].metric("False alarms", f"{int((blocks['alert'] & ~is_anomaly).sum())}")
if analysis.skipped_lines:
    st.warning(
        f"{analysis.skipped_lines:,} lines didn't look like HDFS log lines and were skipped."
    )

st.subheader("Score distribution")
st.altair_chart(score_chart(blocks, bundle.threshold), use_container_width=True)
st.caption("Blocks right of the dashed line are flagged. Hover the bars for counts.")

st.subheader("Blocks, most suspicious first")
explanations = {
    g: explain_block(analysis, bundle, g) for g in blocks.loc[blocks["alert"], "group_id"]
}
blocks["top reason"] = blocks["group_id"].map(
    lambda g: explanations[g][0].text if explanations.get(g) else ""
)
show_all = st.toggle("Show all blocks (not only flagged ones)", value=False)
table = blocks if show_all else blocks[blocks["alert"]]
columns = ["group_id", "score", "n_lines", "top reason"]
if labels is not None:
    columns.insert(3, "true label")
selection = st.dataframe(
    table[columns],
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "group_id": "block",
        "score": st.column_config.NumberColumn("score", format="%.3g"),
        "n_lines": "lines",
    },
)
if table.empty:
    st.success("No block crossed the alert threshold.")
    st.stop()

rows = selection.selection.rows if selection else []
chosen = table.iloc[rows[0]]["group_id"] if rows else table.iloc[0]["group_id"]
st.caption("Click a row to inspect another block.")

# ----------------------------------------------------------------------------- detail
row = blocks.set_index("group_id").loc[chosen]
st.subheader(f"Block {chosen}")
status = "🚩 flagged" if row["alert"] else "not flagged"
extra = f" · true label: **{row['true label']}**" if labels is not None else ""
st.markdown(f"Score **{row['score']:.3g}** (threshold {bundle.threshold:.3g}) · {status}{extra}")

reasons = explanations.get(chosen) or explain_block(analysis, bundle, chosen)
if reasons:
    st.markdown("**Why**")
    st.markdown("\n".join(f"- {KIND_ICON[r.kind]} {escape_markdown(r.text)}" for r in reasons))

left, right = st.columns([2, 3])
with left:
    st.markdown("**Event counts: found vs expected by PCA**")
    st.dataframe(
        expected_vs_found(analysis, bundle, chosen),
        hide_index=True,
        use_container_width=True,
        column_config={"template": st.column_config.TextColumn(width="medium")},
    )
with right:
    st.markdown("**The block's log lines**")
    st.dataframe(block_lines(analysis, chosen), hide_index=True, use_container_width=True)
