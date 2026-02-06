# Workspace Structure (Local)

This file is for local workspace navigation under `Libraries/SAITS`.

## 1) Core Code

- `run_models.py`: main training/testing entry.
- `Simple_RNN_on_imputed_data.py`: downstream RNN script on imputed data.
- `modeling/`: model implementations (`SAITS`, `Transformer`, `BRITS`, `MRNN`).
- `configs/`: experiment config files (`.ini`).
- `NNI_tuning/`: hyperparameter search configs.

## 2) Data Pipeline

- `dataset_generating_scripts/`: dataset download and generation scripts.
- `dataset_generating_scripts/RawData/`: raw downloaded datasets.
- `generated_datasets/`: processed datasets (`datasets.h5`, scaler, logs).

## 3) Experiment Outputs

- `NIPS_results/`: training outputs (checkpoints, logs, tensorboard events).
- `NIPS_results_test/`: test-stage outputs.

## 4) Docs & Reports

- `README.md`: upstream project README.
- `中文文档/`: local Chinese notes/docs.
- `reports/`: organized analysis/report packages.
- `REPORTS.md`: quick pointer to reports.

## 5) Suggested Daily Entry

1. `README.md` for project basics.
2. `WORKSPACE_STRUCTURE.md` for local path mapping.
3. `reports/oil_experiments_2026-02-06/dashboard.html` for oil experiment visual summary.
