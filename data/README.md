# data/

Raw log data lives here. **Everything in this folder except this README is gitignored**:
raw datasets are large, and personal logs must never be committed.

The folder is filled by `logad download` (built in Stage 1), which fetches data only
from official LogHub sources:

- GitHub [`logpai/loghub`](https://github.com/logpai/loghub): 2,000-line samples of each dataset
- Zenodo (the official link is in the `logpai/loghub` README): full datasets (HDFS, BGL, ...)

Expected layout:

```
data/
└── raw/
    ├── hdfs/
    └── bgl/
```
