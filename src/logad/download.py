"""Download raw datasets safely.

The files to fetch are listed in the config's `data.files` section. Safety rules:
- HTTPS only, and only from official LogHub hosts (GitHub raw files and Zenodo),
  also after redirects.
- Every file is checked against a published checksum before it is used. A failed
  download never leaves a half-written file behind.
- Zip archives are checked for "zip-slip" (entries like `../../evil.exe` that would
  write outside the target folder) before anything is extracted.
- Uses only the Python standard library: no extra packages.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ALLOWED_HOSTS = {"raw.githubusercontent.com", "zenodo.org"}
SUPPORTED_HASHES = {"md5", "sha256"}
CHUNK_SIZE = 1024 * 1024  # read 1 MB at a time, so big files never sit fully in memory
USER_AGENT = "logad-downloader/0.1"

Reporter = Callable[[str], None]


def check_url(url: str) -> None:
    """Raise ValueError unless the URL is HTTPS on an allowed host."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only https:// URLs are allowed, got: {url}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"Host {parsed.hostname!r} is not allowed. Allowed: {sorted(ALLOWED_HOSTS)}"
        )


def parse_checksum(checksum: str) -> tuple[str, str]:
    """Split 'md5:abc123' into ('md5', 'abc123')."""
    algorithm, _, expected = checksum.partition(":")
    if algorithm not in SUPPORTED_HASHES or not expected:
        raise ValueError(f"Checksum must look like 'sha256:<hex>' or 'md5:<hex>', got: {checksum}")
    return algorithm, expected.lower()


def file_hash(path: Path, algorithm: str) -> str:
    """Hash a file in chunks (works for files bigger than RAM)."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, checksum: str, report: Reporter = print) -> None:
    """Download `url` to `dest` and verify its checksum.

    Data is first written to `dest.part` and renamed only after the checksum matches,
    so `dest` either is complete and correct, or does not exist.
    """
    check_url(url)
    algorithm, expected = parse_checksum(checksum)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.new(algorithm)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            check_url(response.geturl())  # a redirect must not lead to another host
            total = int(response.headers.get("Content-Length") or 0)
            done, last_percent = 0, -10
            with part.open("wb") as f:
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    percent = done * 100 // total if total else 0
                    if percent >= last_percent + 10:  # progress every 10%
                        report(f"  {dest.name}: {done / 1e6:.1f} MB ({percent}%)")
                        last_percent = percent
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    actual = digest.hexdigest()
    if actual != expected:
        part.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {dest.name}: expected {expected}, got {actual}")
    part.replace(dest)


def safe_extract(zip_path: Path, dest_dir: Path) -> int:
    """Extract a zip archive, refusing any entry that would land outside `dest_dir`.

    Entries already extracted (same file size) are skipped. Returns how many were extracted.
    """
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        # Check every entry BEFORE extracting anything.
        for member in members:
            target = (dest_dir / member.filename).resolve()
            if not target.is_relative_to(dest_dir):
                raise ValueError(f"Unsafe path in {zip_path.name}: {member.filename!r}")

        extracted = 0
        for member in members:
            target = dest_dir / member.filename
            if member.is_dir() or (target.is_file() and target.stat().st_size == member.file_size):
                continue
            archive.extract(member, dest_dir)
            extracted += 1
    return extracted


def download_dataset(config: dict[str, Any], report: Reporter = print) -> list[Path]:
    """Download (and extract) every file in `config['data']['files']` into raw_dir.

    Files that already exist with the right checksum are skipped, so rerunning is cheap.
    Returns the paths of the downloaded files.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    data = config.get("data", {})
    if "generate" in data:  # synthetic dataset: generate it instead of downloading
        from logad.synth import GENERATORS  # local import: only needed for synthetic data

        name = data["generate"]["generator"]
        if name not in GENERATORS:
            raise ValueError(f"Unknown generator {name!r}. Available: {sorted(GENERATORS)}")
        return list(GENERATORS[name](config, report))

    files = data.get("files")
    if not files:
        raise ValueError("Config has no 'data.files' section: nothing to download.")

    paths = []
    for entry in files:
        dest = raw_dir / entry["filename"]
        algorithm, expected = parse_checksum(entry["checksum"])

        if dest.is_file() and file_hash(dest, algorithm) == expected:
            report(f"{dest.name}: already downloaded and verified, skipping.")
        else:
            report(f"{dest.name}: downloading from {urlparse(entry['url']).hostname} ...")
            download_file(entry["url"], dest, entry["checksum"], report)
            report(f"{dest.name}: checksum OK ({algorithm}).")

        if entry.get("extract"):
            report(f"{dest.name}: extracting (skips files already extracted) ...")
            count = safe_extract(dest, raw_dir)
            report(f"{dest.name}: extracted {count} new file(s).")
        paths.append(dest)
    return paths
