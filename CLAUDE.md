# CLAUDE.md: Log Anomaly Detection System

Context file for Claude Code. Read it at the start of every session, and update the
**Progress** section at the end of every session.

Pipeline: LogHub HDFS → BGL → Security logs (SSH auth.log / Windows events).

---

## Who I am
- Learning cybersecurity and data science. Goal: **Security Data Scientist** role.
  This is a flagship portfolio project for job interviews.
- Skill level: basic Python. Trained an ML model once. New to pandas and Jupyter
  (have used Google Colab).
- Machine: Windows 11, 32 GB RAM.
- Goal: build a FULL, professional-quality project FAST (3–4 weeks at 15–20 h/week),
  while understanding it well enough to defend every design decision in an interview.

## Working style: build fast, understand deeply
- **Before each stage:** Claude asks any questions it needs, explains the plan, and
  **waits for me to say "next"**. No files or installs before that.
- **Claude builds each stage completely** (code, tests, docs), following the architecture below.
- **After each stage, Claude explains it:** what it does, why this design, the key lines
  of code, and the trade-offs. Simple English.
- Then **I run it myself**, and Claude asks 2–3 quick questions to check I understand,
  plus 2–3 likely **interview questions** with strong answers.
- One stage per step. Commit after each stage (Conventional Commits:
  `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`).
- Clean, readable code over clever. Don't over-engineer: strong portfolio project,
  not enterprise software.
- Claude runs setup commands itself, explaining each one before running it.

## Machine setup and rules
- **Never install or cache anything on the C: drive.** Everything lives on D:.
  - Python 3.12.10: `D:\Tools\Python312` (project requires 3.12+, because numpy 2.5 does)
  - VS Code 1.145 (portable mode, settings + extensions in `D:\Tools\VSCode\data`):
    `D:\Tools\VSCode`, extensions: Python, Jupyter, Ruff
  - Project: `D:\Projects\log-anomaly-detection` (its own git repo)
  - Virtual env: `.venv/` in the project; its `pip.ini` sets the pip cache to `D:\Tools\pip-cache`
- Notebooks run inside VS Code (ipykernel). Google Colab is the fallback for heavy training
  and hosts a public "Open in Colab" demo notebook.
- Work only inside the project folder. Explain any command that installs software or
  changes system settings before running it.
- Git identity for this repo: `dvolaid-oss <274262795+dvolaid-oss@users.noreply.github.com>`. GitHub: `dvolaid-oss/log-anomaly-detection`
  (public). I create the repo on github.com; Claude pushes only after my OK.
- License: MIT.

---

## Architecture

### Pipeline (data flow)
```
 raw logs ──► [ingest] ──► [parse: Drain] ──► [features] ──► [models] ──► [evaluate] ──► reports/
  (HDFS,        load,        templates +       event-count     baseline,     precision,
   BGL,         normalize     event IDs         matrix per      PCA,          recall, F1,
   auth.log)    to common                      block/window,   IsoForest,    PR curves,
                schema                         TF-IDF option   Autoencoder   comparison
                                                                    │
                                                                    ▼
                                          [explain] ──► Streamlit dashboard (app/)
                                          why was this      upload logs → anomalies
                                          flagged?          + explanations + charts
```

### Design principles
- **Dataset adapters:** each log source (HDFS, BGL, SSH auth.log, Windows events) has an
  adapter that converts it to ONE common schema
  (`timestamp, source, level, message, group_id, label?`). Everything downstream is
  dataset-agnostic, so adding a new log type means writing one new adapter.
- **Common model interface:** every detector implements `fit(X)`, `score(X)`,
  `predict(X, threshold)`, so models are swappable and comparable.
- **Config-driven:** dataset paths, window sizes and model hyperparameters live in
  `configs/*.yaml`, not hardcoded.
- **Unsupervised training:** labels are used ONLY in evaluation, never in training.
- **Reproducible:** fixed random seeds, saved artifacts (parsed data, feature matrices,
  trained models) in `artifacts/`, and a single CLI to rerun everything.
- **Explainability:** every flagged anomaly comes with a reason (e.g. "contains event E22,
  never seen in normal blocks" or "top features driving the score").

### Storage
- No database server. Parsed logs and features are stored as **Parquet** files in `artifacts/`.
- **DuckDB** for SQL queries over the Parquet files (in EDA and the dashboard), to show SQL skills.
- Process large files (full HDFS/BGL) in chunks to stay memory-friendly (target: runs on
  8 GB RAM); fall back to Google Colab if needed.

### Safety rules
- Pin all package versions in `requirements.txt`; only well-known packages; double-check package names.
- Download data only from official LogHub sources (GitHub `logpai/loghub`, Zenodo).
- Never commit raw data, artifacts, or personal logs; `data/` and `artifacts/` are gitignored.
- Security-logs stage: use **synthetic** logs (generator with injected brute-force attacks);
  anonymize anything real.
- Streamlit app: process uploads in memory only, never save them, limit upload size.

### Repo structure
```
log-anomaly-detection/
├── CLAUDE.md                 # context + plan + progress log
├── README.md                 # overview, architecture, results, screenshots, how to run
├── LICENSE                   # MIT
├── pyproject.toml            # package metadata, ruff + pytest config, `logad` entry point
├── requirements.txt          # pinned runtime deps (single source of truth)
├── requirements-dev.txt      # + pytest, ruff, ipykernel
├── configs/                  # hdfs.yaml, bgl.yaml
├── data/                     # raw data (gitignored), download script fills it
├── artifacts/                # parsed logs, features, models (gitignored)
├── docs/                     # setup guide, design notes
├── src/logad/
│   ├── adapters/             # hdfs.py, bgl.py, auth_log.py, windows_events.py
│   ├── parsing/              # drain wrapper, template store
│   ├── features/             # event-count matrix, windowing, TF-IDF
│   ├── models/               # base.py, baseline.py, pca.py, iforest.py, autoencoder.py
│   ├── evaluation/           # metrics, threshold selection, plots
│   ├── explain/              # anomaly explanations
│   ├── config.py             # YAML config loading + validation
│   └── cli.py                # `logad download | parse | features | train | evaluate | run-all`
├── app/dashboard.py          # Streamlit app
├── notebooks/01_exploration.ipynb
├── reports/                  # generated charts + results.md
├── tests/                    # pytest
└── .github/workflows/ci.yml  # ruff + pytest on every push
```

### Tech stack
Python 3.12, pandas, numpy, pyarrow, DuckDB, drain3, scikit-learn, PyTorch (CPU, added in
Stage 6), matplotlib/plotly, Streamlit, PyYAML, typer (CLI), pytest, ruff, GitHub Actions.

---

## Build stages and roadmap

Hours of focused work (build + explanation + running it + questions). Total ≈ 45–65 h,
about 3–4 weeks at 15–20 h/week.

| # | Stage | What gets built | Est. | Week |
|---|-------|-----------------|------|------|
| 0 | Setup | VS Code, venv, full scaffold, CLAUDE.md, CI, README skeleton, GitHub repo | 2–3 h | 1 |
| 1 | Data + EDA | Download script (LogHub 2k samples, later full HDFS from Zenodo), HDFS adapter, EDA notebook | 4–6 h | 1 |
| 2 | Parsing | Drain wrapper, templates saved to artifacts, tests | 3–4 h | 1 |
| 3 | Features | Event-count matrix per block_id, TF-IDF option, tests | 3–4 h | 2 |
| 4 | Models | Model interface, rule baseline, PCA, Isolation Forest | 4–6 h | 2 |
| 5 | Evaluation | Metrics, threshold tuning, PR curves, `reports/results.md` | 4–5 h | 2 |
| 6 | Autoencoder | PyTorch autoencoder, compared against the others | 4–6 h | 3 |
| 7 | Explain + dashboard | Explanations module, Streamlit app, deploy on Streamlit Community Cloud | 6–8 h | 3 |
| 8 | Scale | Full HDFS + BGL adapter with time-window features | 5–8 h | 3–4 |
| 9 | Security twist | Synthetic SSH auth.log (+ maybe Windows events); brute-force / unusual login detection; dashboard demo | 5–8 h | 4 |
| 10 | Polish | README diagram, results, screenshots, dashboard GIF, "what I learned", "future work" (LSTM/DeepLog) | 4–6 h | 4 |

---

## Progress

### Session log
- **2026-11-08:** Discussed the project and chose the setup (see Machine setup). Installed
  VS Code portable on D: (signature verified). Created `.venv` (pip cache on D:), installed and
  pinned dependencies. Scaffolded the repo: package, `logad` CLI (`show-config` works, pipeline
  commands are placeholders), config loader, configs, 9 passing tests, CI, README, setup guide.
  First commit made locally.

### Done
- [x] Stage 0: setup + scaffold (local commit; GitHub push pending)

### Next
- [ ] Me: open the project in VS Code, run the tests myself, answer the check questions
- [ ] Me: create the empty public repo `log-anomaly-detection` on github.com → Claude pushes
- [ ] Stage 1: download script + HDFS adapter + EDA notebook (after I say "next")

### My open questions
- (none yet)

### Decisions log
- Python 3.12 (already installed) instead of 3.11: supported by all our packages,
  PyTorch and Streamlit Community Cloud. The pinned numpy 2.5 requires Python ≥ 3.12, so
  `requires-python = ">=3.12"`.
- VS Code in portable mode so nothing lands on C:.
- Dependencies declared once in `requirements.txt`; `pyproject.toml` reads them dynamically.
- PyTorch deferred to Stage 6 so the base install stays small and CI stays fast.
- Stage 9 uses synthetic security logs: safe to publish, known ground truth.
