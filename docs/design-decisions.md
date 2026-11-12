# Design decisions

Every major choice in this project, why it was made, and what was traded away.
Each entry: **decision** → why → alternative considered.

## Data and evaluation

**Split by time, not at random** (oldest 60% train, next 20% validation, newest 20% test).
→ In production you train on the past and detect in the future. A random split lets a model
see "future" data during training and hides drift. It mattered: HDFS's test period has half
the anomaly rate of training, and Isolation Forest's threshold broke on it (F1 0.825 → 0.501).
→ *Alternative:* random or stratified splits: easier, more stable numbers, but optimistic.

**Unsupervised training: labels are never used to fit a model.**
→ Real logs aren't labelled, and new attack types have no labels yet. Labels are used only to
choose one alert threshold per model (on validation) and to grade (on test).
→ *Alternative:* supervised classifiers would score higher here but can't find unknown
anomalies, and they need labels nobody has in practice.

**Thresholds chosen on the validation split (best F1); each test split used once.**
→ Choosing anything on the test set turns it into a second validation set and inflates the
results. After a test evaluation nothing was tuned; that rule was followed for all three datasets.
→ *Alternative:* fixed percentile thresholds (e.g. top 1% of scores) need no labels at all,
but ignore the very different anomaly rates between datasets.

**Training data keeps its anomalies.** About 3% (HDFS) to 10% (BGL) of training groups are
anomalous. Removing them would require labels, which would no longer be unsupervised.
→ *Cost:* on BGL, frequent alert types look "normal" to the models (19 of 33 test alert types
already occur as alerts in training).

**Report confidence intervals** (bootstrap, 1,000 resamples). A single F1 number can't say
whether two models really differ; non-overlapping intervals can. The bootstrap resamples the
four confusion counts from a multinomial, which is equivalent to resampling rows and far faster.

## Pipeline

**One common schema through adapters** (`timestamp, source, level, component, message,
group_id, label`). Everything after the adapter is dataset-agnostic: adding BGL and SSH logs
meant writing one adapter and one config each, not changing the models or evaluation.

**Config-driven** (`configs/*.yaml`). Paths, masks, windows and model settings live in YAML,
so the same code runs four datasets, and every result can be reproduced from its config.

**Parse with masking + Drain.** Masks remove values we already know are variables (IPs, block
IDs, numbers); Drain groups the rest into templates. Drain is a fast, well-benchmarked parser;
we measured it: 100% grouping accuracy on LogHub's labelled HDFS sample, 96.85% on BGL.
→ *Speed trick:* Drain only sees masked text, so it reads each *distinct* masked message once
(86 for 11.2M HDFS lines) and DuckDB maps the result back. Same groups, seconds not minutes.
→ *Honesty note:* two mask rules (a list of block IDs = one variable; bare 8-digit values =
numbers) were found by scoring on those same labelled samples.

**Event-count features per group** (a block, a time window, or a source IP per window).
The standard representation in log anomaly research: simple, fast, explainable.
→ *Cost:* order, timing and parameter values are lost; this is why account takeover can't be
detected (see the README's Limitations).

**Fixed, non-overlapping windows** for BGL and SSH. Sliding windows overlap the train/test
boundary and leak. Window size (10 minutes) was chosen from training-period statistics only:
bigger windows make "anomalous" the majority (55% at 6 hours), which breaks the "anomalies
are rare" assumption of unsupervised detectors.

**Parquet + DuckDB, no database server.** Columnar files are small and fast; DuckDB runs SQL
over them in-process (EDA, masking 11M lines, joins), with no server to install or secure.

## Models

**Four models behind one interface** (`fit`, `score`, `predict`). Each model owns its feature
transform, so a saved model file is self-contained, and models are swappable and comparable.

**Always a rule baseline** (score = rarity of the block's rarest event). It needs no tuning,
explains itself, and sets the bar: on HDFS it beat Isolation Forest and matched the autoencoder.

**PCA with TF-IDF input** (Xu et al., 2009): the classic HDFS method. Its score is the error
of rebuilding a block from the main "normal" patterns, which also catches **missing** events.
→ *Cost:* deviations *along* the normal patterns (a burst of failed logins) are invisible to
it, which is why it fails on SSH.

**Autoencoder in PyTorch, optional.** A small network (inputs → 32 → 8 → 32 → inputs) as the
non-linear version of PCA, with early stopping on 10% of the *training* data (validation
stays clean). PyTorch is an optional install; for the online app the trained weights are
copied into plain NumPy (same alerts, no 124 MB download).

## Synthetic security data

**Generate SSH logs instead of using real ones.** Real auth logs contain real usernames and
IPs (never publishable) and have no reliable ground truth. The generator is seeded
(reproducible), uses only documentation IP ranges (RFC 5737) and made-up names.

**Background bot scanning is labelled normal.** Every public server is scanned constantly;
a SOC treats it as noise. The real question is finding *targeted* attacks inside that noise.

**The v1 generator flaw was fixed in the data, not hidden.** Version 1's bots never tried
`root`, which made attacks trivially separable (baseline F1 0.934). Version 2's bots behave
like real ones; the new test split was evaluated once (baseline 0.407), and the v1 numbers
stay published in `reports/auth/results_v1_flawed.md`.

## Dashboard

**Each tab uses the model that was best on validation**, never chosen by test results
(HDFS: PCA; SSH: the autoencoder).

**Explanations in plain words,** from three sources: rare events (training frequency), high
counts (training 99th percentile), and missing / extra events (what a rebuilding model
expected). Alerts nobody can understand don't get acted on.

**Safety:** uploads are processed in memory and never stored; uploads are capped at 20 MB;
log text is shown only in tables or escaped markdown; usage telemetry is off.

**Always load this repo's code.** After one deploy the hosted app ran new dashboard code with
old library code still in memory; the dashboard now reloads the `logad` package unless it
loaded it itself from the current files (a test reproduces the original failure).
