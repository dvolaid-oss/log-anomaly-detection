# Setup guide (Windows)

How this project's development environment was set up. Everything lives on the **D:**
drive; nothing is installed or cached on C:.

## 1. Python 3.12

Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/):
choose **Customize installation**, set the install location to `D:\Tools\Python312`, and
tick **Add Python to PATH**.

Check it:

```powershell
python --version      # Python 3.12.x
```

## 2. VS Code (portable, on D:)

1. Download the **ZIP** version (not the installer) from
   [code.visualstudio.com/download](https://code.visualstudio.com/download) → Windows → `.zip` x64.
2. Extract it to `D:\Tools\VSCode`.
3. Create an empty folder `D:\Tools\VSCode\data`. This turns on **portable mode**: settings
   and extensions are stored there instead of in your user folder on C:.
4. Add `D:\Tools\VSCode\bin` to your user PATH so `code .` works from a terminal.
5. Install the extensions **Python**, **Jupyter** and **Ruff**
   (VS Code also suggests them when you open this folder, from `.vscode/extensions.json`).

## 3. Virtual environment

A virtual environment (`.venv`) is a private Python install for this project only, so its
packages never clash with other projects.

```powershell
cd D:\Projects\log-anomaly-detection
python -m venv .venv
```

Keep pip's download cache on D: by creating `.venv\pip.ini`:

```ini
[global]
cache-dir = D:\Tools\pip-cache
```

Activate it and install everything:

```powershell
.venv\Scripts\Activate.ps1          # the prompt now starts with (.venv)
pip install -r requirements-dev.txt
pip install -e . --no-deps          # installs our own package + the `logad` command
```

> If PowerShell refuses to run `Activate.ps1`, you can skip activation and call the venv
> tools directly, e.g. `.venv\Scripts\python -m pytest` and `.venv\Scripts\logad --help`.

## 4. Open the project in VS Code

```powershell
code D:\Projects\log-anomaly-detection
```

- VS Code picks the `.venv` interpreter automatically (from `.vscode/settings.json`).
  Check the bottom-right corner shows `3.12 ('.venv')`.
- **Testing** tab (flask icon) → run all tests.
- Notebooks: open a `.ipynb` file → **Select Kernel** → `.venv`.

## 5. Daily commands

```powershell
logad --help                 # list pipeline commands
pytest                       # run tests
ruff check .                 # lint
ruff format .                # auto-format
```

## 6. GitHub repo

1. On github.com → **New repository** → name `log-anomaly-detection`, **Public**,
   and do **not** add a README, .gitignore or license (we already have them).
2. Connect and push:

```powershell
git remote add origin https://github.com/dvolaid-oss/log-anomaly-detection.git
git push -u origin main
```

Every push then runs the CI workflow (ruff + pytest) under the repo's **Actions** tab.
