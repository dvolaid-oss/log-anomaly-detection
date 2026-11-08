# Log Anomaly Detection

[![CI](https://github.com/dvolaid-oss/log-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/dvolaid-oss/log-anomaly-detection/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An end-to-end, **unsupervised** pipeline that reads system logs, learns what "normal"
looks like, flags anomalies, and **explains why** each one was flagged.

It is benchmarked on the public [LogHub](https://github.com/logpai/loghub) datasets
(HDFS, BGL) and extended to **security logs** (SSH brute-force detection).

> 🚧 **Work in progress.** See the [roadmap](#roadmap) for status.

## How it works

```
 raw logs ──► [ingest] ──► [parse: Drain] ──► [features] ──► [models] ──► [evaluate] ──► reports/
  (HDFS,        load,        templates +       event-count     baseline,     precision,
   BGL,         normalize     event IDs         matrix per      PCA,          recall, F1,
   auth.log)    to common                      block/window,   IsoForest,    PR curves,
                schema                         TF-IDF option   Autoencoder   comparison
                                                                    │
                                                                    ▼
                                          [explain] ──► Streamlit dashboard
```

1. **Adapters** convert each log format into one common schema
   (`timestamp, source, level, message, group_id, label`), so everything downstream
   is dataset-agnostic.
2. **Drain** turns millions of raw lines into a few dozen event templates.
3. **Features:** event counts per session (HDFS block) or time window (BGL).
4. **Models** are trained without labels: a rule baseline, PCA, Isolation Forest,
   and an autoencoder, all behind one `fit / score / predict` interface.
5. **Evaluation** uses the labels only here: precision, recall, F1, PR curves.
6. **Explanations** say why each anomaly was flagged, shown in a Streamlit dashboard.

## Data

[LogHub HDFS_v1](https://github.com/logpai/loghub): 11.2M log lines from a Hadoop cluster,
grouped into 575,061 blocks (sessions), of which **2.93% are labelled anomalous**.
Downloaded by `logad download` from Zenodo with checksum verification (see [data/README.md](data/README.md)).

Exploratory analysis: [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb). Key findings:

- Only INFO/WARN levels: anomalies must be found from patterns, not error messages.
- A "flag any WARN" rule fails: 48% of anomalous blocks have a WARN, but so do 23% of normal ones.
- Short blocks are suspicious: no normal block has under 13 lines, but 37% of anomalous blocks do.
- Some message types are strong signals (e.g. one appears in 30% of anomalous vs 0.04% of normal blocks).
- ~10M distinct messages reduce to under 100 templates → a log parser is essential.

![Block length, normal vs anomalous](reports/hdfs/eda_block_length.png)

## Parsing

`logad parse` turns each message into a template (event ID): regex **masking** of known
variables (block IDs, IPs, numbers), then the **Drain** algorithm groups what's left.

| | Result |
|---|---|
| Accuracy on LogHub's hand-labelled 2k sample | **100% grouping accuracy**, 14 of 14 templates |
| Full HDFS: 11.2M lines (10.3M distinct messages) | **45 templates in ~7 s** |

Speed trick: Drain only sees the masked text, so it reads each *distinct* masked message once
(86 for HDFS) and DuckDB maps the results back to all 11.2M lines. LogHub's reference list has
29 templates; our extra ones are rare Java-exception variants (< 0.1% of lines each).
Full table: [reports/hdfs/templates.md](reports/hdfs/templates.md).

## Features

`logad features` turns each block into a row of event counts: a **575,061 × 45 matrix**,
built in ~8 s. Blocks are split **by time** so models learn from the past and are judged on
the future:

| Split | Blocks | Anomaly % | Used for |
|-------|-------:|----------:|----------|
| train (oldest 60%) | 345,037 | 3.23 | fitting models (labels never read) |
| validation (next 20%) | 115,012 | 3.48 | choosing alert thresholds |
| test (newest 20%) | 115,012 | 1.46 | final, untouched evaluation |

The anomaly rate drops in the test period: a realistic shift that makes precision harder.
Optional TF-IDF weighting (rare events count more) and scaling are learned on the train split only.

## Results

_Coming in Stage 5._

## Quickstart (Windows, PowerShell)

```powershell
git clone https://github.com/dvolaid-oss/log-anomaly-detection.git
cd log-anomaly-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e . --no-deps

logad --help          # list pipeline commands
pytest                # run the tests

logad download --config configs/hdfs.yaml   # 187 MB from Zenodo, checksum-verified
logad ingest   --config configs/hdfs.yaml   # raw logs → artifacts/hdfs/logs.parquet (~1 min)
logad parse    --config configs/hdfs.yaml   # messages → templates, artifacts/hdfs/events.parquet
logad features --config configs/hdfs.yaml   # event-count matrix per block + time-based splits
```

Full setup guide: [docs/setup-windows.md](docs/setup-windows.md).

## Project structure

```
configs/          dataset configs (paths, window sizes, hyperparameters)
src/logad/        the Python package: adapters, parsing, features, models, evaluation, explain, cli
app/              Streamlit dashboard
notebooks/        exploratory data analysis
tests/            pytest suite
data/, artifacts/ raw data and generated files (not committed)
```

## Roadmap

| # | Stage | Status |
|---|-------|--------|
| 0 | Setup, scaffold, CI | ✅ |
| 1 | Data download + HDFS adapter + EDA | ✅ |
| 2 | Parsing with Drain | ✅ |
| 3 | Features (event counts, TF-IDF) | ✅ |
| 4 | Models: baseline, PCA, Isolation Forest | ⏳ |
| 5 | Evaluation + results table | ⏳ |
| 6 | Autoencoder (PyTorch) | ⏳ |
| 7 | Explanations + Streamlit dashboard | ⏳ |
| 8 | BGL (time windows) | ⏳ |
| 9 | Security logs: SSH brute-force detection | ⏳ |
| 10 | Polish | ⏳ |

## License

[MIT](LICENSE)
