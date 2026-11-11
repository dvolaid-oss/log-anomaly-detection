# Deploying the dashboard (Streamlit Community Cloud, free)

**Live app:** https://log-anomaly-detection-nzno5mxmwpjotewlg6eyqm.streamlit.app/

The dashboard (`app/dashboard.py`) runs online straight from this GitHub repo. Everything it
needs is committed: one model bundle per tab (`app/model/hdfs/`, `app/model/auth/`, a few
KB each) and the demo logs (`app/demo/`).
No dataset download, no PyTorch.

## Run it locally first

```powershell
.venv\Scripts\Activate.ps1
streamlit run app/dashboard.py
```

It opens at http://localhost:8501. Stop it with `Ctrl+C` in the terminal.

## Deploy (about 5 minutes, done by the repo owner)

1. Go to **https://share.streamlit.io** and **sign in with GitHub** (account `dvolaid-oss`).
   Streamlit asks for permission to read your repos: that's how it gets the code.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `dvolaid-oss/log-anomaly-detection`
   - **Branch:** `main`
   - **Main file path:** `app/dashboard.py`
   - **App URL:** pick a name, e.g. `log-anomaly-detector`
4. Open **Advanced settings** → **Python version: 3.12** (the project needs ≥ 3.12).
5. Click **Deploy**. The first start installs `requirements.txt` and takes a few minutes.
6. Copy the app's URL and put it in the README's "Live demo" line.

## Notes

- **Updates:** every push to `main` redeploys automatically. After the Stage 9 push, the
  hosted app ran the NEW dashboard with OLD `logad` code (even after a reboot), so the
  dashboard now always loads `src/` from this repo and drops stale `logad` modules
  (tested). If the app still misbehaves after a push: **Manage app** (bottom right) → **⋮**
  → **Reboot app**.
- **Settings** come from `.streamlit/config.toml` (20 MB upload limit, no usage telemetry).
- **Sleep:** free apps sleep after a few days without visitors; the first visit wakes them
  (about 30 seconds).
- **Privacy:** uploaded logs are processed in memory and never stored, but they are sent to
  Streamlit's servers. Don't upload private company logs to the public app; run it locally
  for those.
- **Updating a model:** rerun the pipeline, then `logad export-app --config configs/hdfs.yaml`
  (or `configs/auth.yaml`) and commit `app/model/` and `app/demo/`.
