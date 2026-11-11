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

## Session rule: continue until ~90% context, then hand off
(Changed 2026-11-08: was "one stage per session".) Stages continue in the **same** Claude
Code session until its context is about **90% used**. Then Claude updates **Progress →
Hand-off** here and tells me to open a new session in `D:\Projects\log-anomaly-detection`
(so this file loads automatically). Claude can't see an exact percentage, so it estimates and
hands off early rather than late, ideally at the end of a stage. This file is the only memory
that carries between sessions, so it must always be complete and up to date.

Session flow:
1. Read this file, especially **Progress → Hand-off for the next session**.
2. Finish anything left open from the previous stage (e.g. my unanswered check questions).
3. Ask me any questions → explain the plan → **wait for "next"**.
4. Build the stage → explain it → I run it → check questions + interview questions.
5. Commit, update **Progress** (session log, done, hand-off), push after my OK.
6. Tell me the stage is finished; continue with the next stage here unless context is near 90%.

Kickoff message for a new session:
> Read CLAUDE.md and continue from the hand-off following the session rule.

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
- Git identity for this repo: `dvolaid-oss <274262795+dvolaid-oss@users.noreply.github.com>`
  (GitHub noreply address; my GitHub account blocks pushes that expose my real email).
  **Never write my real email anywhere in the repo.**
- GitHub: `dvolaid-oss/log-anomaly-detection`, **public**. Claude pushes only after my OK.
  Before each push, check that no private data (emails, personal paths, secrets, logs) is
  being committed: the whole history is public.
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
├── configs/                  # hdfs.yaml (full), hdfs_sample.yaml (2k lines), bgl.yaml
├── data/                     # raw data (gitignored), download script fills it
├── artifacts/                # parsed logs, features, models (gitignored)
├── docs/                     # setup guide, design notes
├── src/logad/
│   ├── adapters/             # base.py (common schema), hdfs.py, bgl.py, auth_log.py, ...
│   ├── parsing/              # drain.py (masking + Drain wrapper), accuracy.py (grouping accuracy), __init__ (pipeline)
│   ├── features/             # __init__ (event-count matrix + time splits), transform.py (TF-IDF, scaling)
│   ├── models/               # base.py, baseline.py, pca.py, iforest.py, (autoencoder.py), __init__ (registry + train_models)
│   ├── evaluation/           # metrics.py (thresholds, P/R/F1, bootstrap CI), plots.py, __init__ (evaluate_scores + results.md)
│   ├── explain/              # anomaly explanations
│   ├── config.py             # YAML config loading + validation
│   ├── download.py           # safe downloads: host allowlist, checksums, zip-slip check
│   └── cli.py                # `logad download | ingest | parse | features | train | evaluate | run-all`
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
| 1 | Data + EDA | Download script (2k sample + full HDFS_v1 from Zenodo), HDFS adapter, `ingest`, EDA notebook | 4–6 h | 1 |
| 2 | Parsing | Drain wrapper, templates saved to artifacts, tests | 3–4 h | 1 |
| 3 | Features | Event-count matrix per block_id, TF-IDF option, tests | 3–4 h | 2 |
| 4 | Models | Model interface, rule baseline, PCA, Isolation Forest | 4–6 h | 2 |
| 5 | Evaluation | Metrics, threshold tuning, PR curves, `reports/results.md` | 4–5 h | 2 |
| 6 | Autoencoder | PyTorch autoencoder, compared against the others | 4–6 h | 3 |
| 7 | Explain + dashboard | Explanations module, Streamlit app, deploy on Streamlit Community Cloud | 6–8 h | 3 |
| 8 | Scale | BGL adapter with time-window features (full HDFS moved to Stage 1) | 5–8 h | 3–4 |
| 9 | Security twist | Synthetic SSH auth.log (+ maybe Windows events); brute-force / unusual login detection; dashboard demo | 5–8 h | 4 |
| 10 | Polish | README diagram, results, screenshots, dashboard GIF, "what I learned", "future work" (LSTM/DeepLog) | 4–6 h | 4 |

---

## Progress

### Session log
- **2026-11-08:** Discussed the project and chose the setup (see Machine setup). Installed
  VS Code portable on D: (signature verified). Created `.venv` (pip cache on D:), installed and
  pinned dependencies. Scaffolded the repo: package, `logad` CLI (`show-config` works, pipeline
  commands are placeholders), config loader, configs, 9 passing tests, CI, README, setup guide.
  I ran the tests and CLI myself in VS Code (9 passed). Pushed to GitHub; first push was
  rejected (GH007, email privacy) → switched to the GitHub noreply email. Repo made public.
  Introduced the one-stage-per-session rule.
- **2026-11-08 (Stage 1):** I postponed answering the Stage 0 check questions ("ask later") and
  said to continue with Claude's recommended defaults. Built: `logad download` (allowlisted
  hosts, md5/sha256 checks, atomic `.part` files, zip-slip check, stdlib only), common schema
  (`adapters/base.py`), HDFS adapter + `logad ingest` (chunked, 1M lines per chunk → Parquet),
  `configs/hdfs_sample.yaml`, EDA notebook, 23 new tests (32 total, all pass, ruff clean).
  Claude verified on real data: full ingest = 11,175,629 lines, 575,061 blocks, 16,838 anomalous
  (2.93%), 0 skipped lines, ~65 s, peak RAM 1.9 GB, Parquet 187 MB. Notebook ran end-to-end (~13 s).
  I ran download, ingest and the notebook myself (after fixing two kernel issues: wrong kernel
  picked, and a DuckDB progress-bar setting that fails in Jupyter) and saved it with outputs.
  Claude explained all 6 check questions (Stage 0 + Stage 1); I said "questions later", so
  my own answers and the interview questions are still open. Committed and pushed Stage 1.
  Changed the session rule: keep going in the same session until ~90% context.
- **2026-11-08 (Stage 2, same session):** Built masking + Drain parsing: `parsing/drain.py`
  (`LogParser`: fit / match / templates / save / load), `parsing/accuracy.py` (grouping
  accuracy), `parse_dataset` pipeline, `logad parse`; masks in configs; ground-truth file
  added to `hdfs_sample.yaml`; 14 new tests (46 total). Results: 2k sample 100% grouping
  accuracy (14/14 templates; was 99.75% before the BLK-list mask); full HDFS 11.2M lines →
  86 distinct masked messages → 45 templates in ~7 s. Outputs: `artifacts/hdfs/events.parquet`
  (line_id, timestamp, group_id, event_id, label), `templates.csv`, `drain_parser.json.zlib`,
  `reports/<dataset>/templates.md`. I ran download + both parse commands myself (same results).
  Committed and pushed Stage 2.
- **2026-11-08 (Stage 3, same session):** I said I watch the context meter myself and will
  tell Claude when to hand off (it was 24% when Claude guessed 60%). Built `logad features`:
  event-count matrix per block (`features/__init__.py`), time-based train/validation/test
  split 60/20/20, `features/transform.py` (TF-IDF weighting + optional StandardScaler, sklearn
  API, fitted on train only), `reports/<dataset>/features.md`, 13 new tests (59 total).
  Full HDFS: 575,061 × 45 in ~8 s; anomaly rate train 3.23%, validation 3.48%, test 1.46%.
  I ran `logad features` myself (same results). Committed and pushed Stage 3.
- **2026-11-08 (Stage 4, same session):** Built `models/`: `AnomalyDetector` base (owns its
  transformer; fit/score/predict/save/load via joblib), `RareEventDetector` (rule baseline),
  `PCADetector` (SPE, TF-IDF input), `IsolationForestDetector`; `train_models` + `logad train`
  (fit on train split, save `artifacts/<ds>/models/*.joblib` + `scores.parquet` for all
  blocks, quick validation ROC-AUC/PR-AUC). 17 new tests (76 total). Full HDFS, validation:
  baseline ROC 0.889 / PR 0.810, PCA 0.981 / 0.859 (5 components), IForest 0.987 / 0.807;
  each fits in < 1 s. Test split NOT looked at yet. I ran `logad train` myself (same numbers).
  Committed and pushed Stage 4.
- **2026-11-09 (Stage 5, same session, next morning):** Built `evaluation/`: best-F1 threshold
  on validation, test metrics once, bootstrap 95% CIs (multinomial trick), PR-curve and
  score-distribution charts (dataviz palette validated: blue/orange/aqua, aqua needs labels),
  `thresholds.json`, `reports/hdfs/results.md`; `logad evaluate`, `logad run-all`; all CLI
  commands show one-line errors (`friendly_errors`); `predict` now alerts on score >= threshold.
  10 new tests (86 total). **TEST SPLIT EVALUATED ONCE on 2026-11-09:** F1 baseline 0.719
  [0.700–0.735], PCA 0.980 [0.975–0.985] (precision 0.997, recall 0.964, 5 false alarms),
  IForest 0.501 [0.486–0.516] (validation F1 0.825 → threshold didn't transfer; 2,673 FAs).
  VS Code wasn't open; Claude started it (`D:\Tools\VSCode\Code.exe <project>`). I ran
  `logad evaluate` myself (worked). Committed and pushed Stage 5.
- **2026-11-11 (Stage 6, same session):** I paused once ("wait") before the install, then said
  to continue. Installed PyTorch 2.14.1+cpu into `.venv` (124 MB download, 536 MB on D:;
  optional `requirements-torch.txt`, CI installs it). Built `models/autoencoder.py`
  (45→32→8→32→45, ReLU, MSE, Adam, early stopping on 10% of TRAIN, lazy torch import);
  autoencoder added to `configs/hdfs.yaml` only (sample config stays torch-free). 6 new tests
  (92 total). Validation: ROC 0.996 / PR 0.872 (best), 16 s, 29 epochs. **Test evaluated once
  (2026-11-11):** F1 0.714 [0.696–0.731], precision 0.705, recall 0.724; the 3 existing rows
  are byte-identical to before. Analysis (no tuning): all 463 missed anomalies are the same
  2-line block (E1 + E2: write died after allocation); AE and baseline both miss exactly these,
  PCA catches all 463 (fixed-proportion structure → notices missing events).
  VS Code was closed again; Claude launched it for me. I ran `logad train` + `logad evaluate`
  myself (same numbers). Committed and pushed Stage 6.

### Done
- [x] Stage 0: setup + scaffold, pushed to GitHub (public)
- [x] Stage 1: download + HDFS adapter + ingest + EDA notebook, pushed (check questions postponed)
- [x] Stage 2: masking + Drain parsing (100% grouping accuracy on 2k sample), pushed
- [x] Stage 3: event-count matrix + time splits + TF-IDF/scaling transformer, pushed
- [x] Stage 4: model interface + baseline, PCA, Isolation Forest + `logad train`, pushed
- [x] Stage 5: evaluation (thresholds on validation, test once, bootstrap CIs, charts, run-all), pushed
- [x] Stage 6: PyTorch autoencoder (test F1 0.714; PCA still best at 0.980), pushed

### Hand-off for the next session
- **Next: Stage 7** (explanations module + Streamlit dashboard + deploy on
  Streamlit Community Cloud; use PCA as the main model, no torch needed online).
- Deleted (with my OK, 2026-11-11) an old 16.6 MB pip cache on C: (`%LOCALAPPDATA%\pip\cache`,
  from 2026-11-08, before the D: cache setting).
- If VS Code is closed when I need to run something, Claude launches it:
  `Start-Process D:\Tools\VSCode\Code.exe "D:\Projects\log-anomaly-detection"`.
- **RULE: the HDFS test split has been used.** Never tune anything (models, thresholds,
  features, masks) based on test results. Any new model (e.g. Stage 6 autoencoder) picks its
  threshold on validation and is evaluated on test once; report it as an addition, and never
  re-tune the existing models after seeing the test numbers.
- Still open, ask me when I want ("questions later"):
  - My own answers to the 6 check questions (Claude already explained the answers):
    Stage 0: (1) why `data/` holds only a README, (2) why `show-config` paths differ per
    dataset, (3) what `.venv` is for. Stage 1: (4) why `.part` files, (5) the "flag every WARN"
    rule (recall ~48%, precision ~6%), (6) why labels next to the data is OK (training never
    reads them; otherwise data leakage).
  - Stage 1 and Stage 2 interview questions with strong answers (not given yet), and
    Stage 2–6 check questions (not asked yet).
- CI status (public repo):
  `curl -s https://api.github.com/repos/dvolaid-oss/log-anomaly-detection/actions/runs?per_page=3`.
- Data on my machine: `data/raw/hdfs` (full, 1.6 GB) and `data/raw/hdfs_sample`; parsed
  `artifacts/hdfs/logs.parquet` (11.2M rows). Rerun with `logad download` / `logad ingest`.
- Things I already know how to do: open the project in VS Code, open the terminal
  (View → Terminal), activate the venv (`.venv\Scripts\Activate.ps1`), run `pytest` and `logad`,
  open a notebook, pick the `.venv` kernel, Run All, save.

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
- Sessions: originally one stage per session; changed on 2026-11-08 to "continue in the same
  session until ~90% context, then hand off via this file".
- Stage 1 downloads the **full HDFS_v1** (not only the 2k sample), because the 2k sample has
  no labels. The sample stays as `configs/hdfs_sample.yaml` for quick runs. Stage 8 is now BGL only.
- Two configs (`hdfs.yaml`, `hdfs_sample.yaml`) instead of a `--variant` flag: each config is
  one dataset with its own paths, so later stages need no special cases.
- Common schema = `line_id, timestamp, source, level, component, message, group_id, label`
  (added `line_id` and `component` to the original plan). One row per (line, block): a line
  mentioning several blocks is repeated per block, as in the loglizer reference code.
  `label`: 1 anomaly, 0 normal, null unknown; used only for evaluation.
- New `logad ingest` step (adapter → `artifacts/<dataset>/logs.parquet`), separate from `parse`.
- Downloads use only the standard library (`urllib`, `hashlib`, `zipfile`): no new packages.
- Notebook outputs are committed so charts render on GitHub; charts saved to `reports/hdfs/`.
- Notebooks: test them in a real Jupyter kernel (jupyter_client), not as a plain script. In
  Jupyter, DuckDB 2.0 refuses `SET enable_progress_bar` without ipywidgets, so we don't set it.
- Parsing: mask first, then run Drain on DISTINCT masked messages only (86 instead of 11.2M):
  same groups because Drain only sees masked text; ~7 s instead of minutes. DuckDB does the
  bulk masking (RE2 regex), Python `re` masks single new messages: a test checks both agree.
- Drain is fitted on all lines (no labels involved; in production it runs on all incoming logs).
  Stricter "fit on training split only" can be revisited in Stage 4.
- BLK mask also swallows a LIST of block IDs (`blk_1 blk_2 ...` = one variable). Found by
  scoring on the 2k ground truth (99.75% → 100%). It's a general rule, but it was tuned on the
  same sample we score on: say so honestly if asked.
- Full HDFS: 45 templates vs LogHub's 29 reference templates. The extra ones are rare
  Java-exception variants with different token counts (Drain only groups equal-length messages).
- Saved parser = zlib-compressed JSON with a jsonpickle'd Drain tree: only load files we made
  (jsonpickle can rebuild arbitrary objects).
- DuckDB 2.0: relations have no `fetchone()`; use `con.sql(q).fetchall()`.
- Features: split by TIME (block's first line), 60/20/20 train/validation/test, not random:
  train on the past, judge on the future (no leakage backwards in time). Validation is for
  threshold choice (Stage 5), test is touched once for final numbers.
- Training data keeps its ~3% anomalous blocks (removing them would need labels → not
  unsupervised).
- TF-IDF uses sklearn's smoothed IDF `ln((1+N)/(1+df)) + 1` (every weight ≥ 1; an event in
  every block gets 1.0). TF-IDF and scaling are fitted on train only.
- `features.parquet` stores RAW counts; transformations happen at model time, so one
  matrix serves every model/config.
- Test split has a lower anomaly rate (1.46% vs ~3.3%): time drift, reported, not "fixed".
- Models: each detector OWNS its transformer (fitted with it on train), so a saved model file
  is self-contained. Per-model `transform` in config overrides `features.type` (PCA: tfidf).
- Rule baseline = rarest-event rarity `-log10(share of train blocks with the event)`; events
  never seen get share 1/(n+1). Gives a free explanation (which event) for Stage 7.
- PCA score = squared prediction error (SPE); threshold chosen in Stage 5, not the Q-statistic.
- Isolation Forest can't see events constant in training (never seen). In HDFS, E44 and E45
  never occur in train; all 8 val/test blocks with them are anomalous. Baseline/PCA catch them.
- Quick check after training = validation ROC-AUC + PR-AUC (threshold-free). Labels read only
  for reporting. The test split is not looked at until Stage 5's final evaluation.
- Models saved with joblib (pickle): only load files this project created.
- Evaluation: threshold = best F1 on VALIDATION (alert when score >= threshold); test used ONCE.
  Labels influence the system only through these thresholds (one number per model).
- Bootstrap CIs via multinomial resampling of the confusion counts (equivalent to resampling
  blocks, much faster). Unpaired per-model intervals: non-overlap = clearly different.
- Isolation Forest's validation→test drop is reported as a finding (threshold fragility under
  time drift), NOT fixed by re-tuning on test.
- Charts: palette blue #2a78d6 / orange #eb6834 / aqua #1baf7a (fixed order, validated for
  colour-blindness); line styles + direct labels as second cue; text in neutral ink.
  Stage 6 added slot 4 yellow #eda100 (4-colour set re-validated). More models than 4 → redesign.
- PyTorch is OPTIONAL (`requirements-torch.txt`, `--extra-index-url` CPU index, pinned
  `torch==2.14.1+cpu`; chose 2.14.1 over brand-new 2.15.0 for stability). The autoencoder
  module imports torch lazily, so everything else runs without it.
- Autoencoder: tfidf + StandardScaler input; early stopping on 10% of TRAIN (validation stays
  clean for thresholds); score = per-block MSE. Unlike Isolation Forest it DOES flag events
  never seen in training.
- Finding from test analysis (not used for tuning): "missing-event" anomalies (2-line blocks,
  E1+E2) fool the AE and the baseline but not PCA. Any fix (e.g. a block-length or
  expected-events feature) must be validated on new data (BGL, Stage 8), not on this test set.
- VS Code's kernel picker labels the global `D:\Tools\Python312` "Recommended": always pick `.venv`.
