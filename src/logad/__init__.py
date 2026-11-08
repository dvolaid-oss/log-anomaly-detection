"""logad: unsupervised log anomaly detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Read the version from the installed package metadata (pyproject.toml is the source).
    __version__ = version("logad")
except PackageNotFoundError:  # running from source without `pip install -e .`
    __version__ = "0.0.0+unknown"
