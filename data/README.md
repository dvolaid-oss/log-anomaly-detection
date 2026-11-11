# data/

Raw log data lives here. **Everything in this folder except this README is gitignored**:
raw datasets are large, and personal logs must never be committed.

The folder is filled by `logad download`, which fetches data only from official LogHub
sources over HTTPS and checks every file against a published checksum:

| Config | Source | Download | Labels |
|--------|--------|----------|--------|
| `configs/hdfs_sample.yaml` | GitHub [`logpai/loghub`](https://github.com/logpai/loghub), pinned commit | `HDFS_2k.log`, 288 KB, 2,000 lines + `HDFS_2k.log_structured.csv` (true template per line, to score the parser) | no |
| `configs/hdfs.yaml` | [Zenodo record 8196385](https://zenodo.org/records/8196385) | `HDFS_v1.zip`, 187 MB → 1.6 GB, 11.2M lines | yes, per block |
| `configs/bgl_sample.yaml` | GitHub `logpai/loghub`, pinned commit | `BGL_2k.log` + `BGL_2k.log_structured.csv` (true templates) | yes, per line |
| `configs/bgl.yaml` | [Zenodo record 8196385](https://zenodo.org/records/8196385) | `BGL.zip`, 57 MB → 0.7 GB, 4.7M lines | yes, per line (first word) |

```powershell
logad download --config configs/hdfs.yaml   # download + verify + extract
logad ingest   --config configs/hdfs.yaml   # → artifacts/hdfs/logs.parquet
```

Layout after downloading:

```
data/
└── raw/
    ├── hdfs_sample/
    │   ├── HDFS_2k.log
    │   └── HDFS_2k.log_structured.csv      ← ground-truth templates (parser accuracy)
    └── hdfs/
        ├── HDFS_v1.zip
        ├── HDFS.log                        ← used by the HDFS adapter
        └── preprocessed/
            ├── anomaly_label.csv           ← BlockId,Label (Normal/Anomaly), evaluation only
            └── ...                         (LogHub's own parsed versions, not used)
```
