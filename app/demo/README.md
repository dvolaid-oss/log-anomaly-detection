# Demo data

`hdfs_demo.log` holds the raw log lines of **300 HDFS blocks** (4,320 lines, 0.6 MB), and
`hdfs_demo_labels.csv` holds their Normal/Anomaly labels. The dashboard uses them for its
"Try the demo" mode, so visitors without HDFS logs can still see the model work.

- **Source:** the public HDFS_v1 dataset from [LogHub](https://github.com/logpai/loghub)
  ([Zenodo record 8196385](https://zenodo.org/records/8196385)), originally from
  W. Xu et al., *Detecting Large-Scale System Problems by Mining Console Logs*, SOSP 2009.
  LogHub: J. Zhu, S. He, P. He, J. Liu, M. R. Lyu, *Loghub: A Large Collection of System Log
  Datasets for AI-driven Log Analytics*, ISSRE 2023. According to the LogHub README, the
  datasets are freely available for research and academic work.
- **Selection:** 270 normal + 30 anomalous blocks drawn at random (seed 42) from the **test
  split**, i.e. blocks the model never trained on. Lines are copied unchanged, so the demo
  shows the model's honest behaviour, including the anomalies it misses.
- **Why it's committed:** this project never commits raw data, with this one deliberate
  exception: it's tiny, public research data with credit, and contains nothing personal
  (hosts are private 10.x addresses from a 2008 research cluster).
- **Rebuild it:** `logad export-app --config configs/hdfs.yaml` (needs the full dataset).
