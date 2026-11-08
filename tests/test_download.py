import hashlib
import io
import zipfile

import pytest

from logad import download
from logad.download import check_url, download_dataset, download_file, parse_checksum, safe_extract

CONTENT = b"081109 203615 148 INFO dfs.DataNode: hello blk_1\n"
GOOD_URL = "https://raw.githubusercontent.com/logpai/loghub/abc/HDFS/HDFS_2k.log"


class FakeResponse(io.BytesIO):
    """Stands in for urlopen's response, so tests never touch the network."""

    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}
        self._url = url

    def geturl(self):
        return self._url


def fake_urlopen(data: bytes, final_url: str = GOOD_URL):
    return lambda request, timeout: FakeResponse(data, final_url)


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "url",
    [
        "http://raw.githubusercontent.com/x",  # not HTTPS
        "https://evil.example.com/HDFS.zip",  # not an allowed host
        "https://zenodo.org.evil.com/HDFS.zip",  # look-alike host
        "file:///etc/passwd",
    ],
)
def test_check_url_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        check_url(url)


def test_check_url_accepts_official_hosts():
    check_url(GOOD_URL)
    check_url("https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1")


def test_parse_checksum():
    assert parse_checksum("md5:ABC") == ("md5", "abc")
    with pytest.raises(ValueError):
        parse_checksum("crc32:abc")


def test_download_file_verifies_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", fake_urlopen(CONTENT))
    dest = tmp_path / "HDFS_2k.log"
    download_file(GOOD_URL, dest, sha256(CONTENT), report=lambda msg: None)
    assert dest.read_bytes() == CONTENT


def test_download_file_bad_checksum_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", fake_urlopen(CONTENT))
    dest = tmp_path / "HDFS_2k.log"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file(GOOD_URL, dest, sha256(b"something else"), report=lambda msg: None)
    assert list(tmp_path.iterdir()) == []  # no file, no leftover .part


def test_download_file_rejects_redirect_to_other_host(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", fake_urlopen(CONTENT, "https://evil.example.com/x"))
    with pytest.raises(ValueError, match="not allowed"):
        download_file(GOOD_URL, tmp_path / "f.log", sha256(CONTENT), report=lambda msg: None)
    assert list(tmp_path.iterdir()) == []


def test_download_dataset_skips_verified_file(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("should not download a file that is already verified")

    monkeypatch.setattr(download, "urlopen", no_network)
    (tmp_path / "HDFS_2k.log").write_bytes(CONTENT)
    config = {
        "paths": {"raw_dir": str(tmp_path)},
        "data": {
            "files": [{"url": GOOD_URL, "filename": "HDFS_2k.log", "checksum": sha256(CONTENT)}]
        },
    }
    messages = []
    download_dataset(config, report=messages.append)
    assert "skipping" in messages[0]


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_safe_extract_extracts_then_skips_existing(tmp_path):
    zip_path = tmp_path / "data.zip"
    make_zip(zip_path, {"HDFS.log": "log", "preprocessed/anomaly_label.csv": "BlockId,Label\n"})
    out = tmp_path / "out"
    assert safe_extract(zip_path, out) == 2
    assert (out / "preprocessed" / "anomaly_label.csv").is_file()
    assert safe_extract(zip_path, out) == 0  # already there → nothing re-extracted


def test_safe_extract_blocks_zip_slip(tmp_path):
    zip_path = tmp_path / "evil.zip"
    make_zip(zip_path, {"ok.txt": "fine", "../../evil.txt": "pwned"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(zip_path, out)
    assert not (tmp_path / "evil.txt").exists()
    assert not (out / "ok.txt").exists()  # nothing extracted at all
