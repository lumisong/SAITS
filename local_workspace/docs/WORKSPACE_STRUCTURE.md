# Workspace Structure (Partitioned)

This file explains the partitioned layout under `Libraries/SAITS`.

## 1) Upstream Source Area (Root)

- `README.md`, `run_models.py`, `modeling/`, `configs/`, `dataset_generating_scripts/`, `NNI_tuning/`
- This area is reserved for upstream-compatible source code.

## 2) Local Workspace Area

- `local_workspace/docs/`: local notes, structure docs, Chinese docs.
- `local_workspace/reports/`: experiment reports and dashboards.
- `local_workspace/configs/`: your custom experiment config files.
- `local_workspace/dataset_scripts/`: your custom data scripts/notebooks.
- `local_workspace/scripts/`: local run scripts.
- `local_workspace/sandbox/`: temporary test files.
- `local_workspace/assets/`: local extra figures/assets.

## 3) Local Runtime Data

- `local_workspace/data/RawData/`: raw datasets.
- `local_workspace/data/generated_datasets/`: processed datasets.
- `local_workspace/artifacts/NIPS_results/`: training outputs/checkpoints.
- `local_workspace/artifacts/NIPS_results_test/`: testing outputs.

## 4) Daily Entry

1. `README.md` for upstream project usage.
2. `LOCAL_WORKSPACE.md` for local experiment navigation.
3. `local_workspace/reports/oil_experiments_2026-02-06/dashboard.html` for visual report.
