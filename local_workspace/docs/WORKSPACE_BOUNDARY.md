# Workspace Boundary (After Physical Partition)

Updated: 2026-02-06

This repository is now split into two physical zones:

## 1) Upstream-Compatible Source Zone

Path: `Libraries/SAITS/` root

Typical entries:

- `.github/`
- `README.md`
- `run_models.py`
- `configs/`
- `dataset_generating_scripts/`
- `modeling/`
- `NNI_tuning/`
- `CITATION.cff`, `LICENSE`, `conda_env_dependencies.yml`

Rule: keep this zone close to upstream for easier future sync.

## 2) Local Experiment Workspace Zone

Path: `Libraries/SAITS/local_workspace/`

Local-only areas:

- `docs/`
- `reports/`
- `configs/`
- `dataset_scripts/`
- `scripts/`
- `sandbox/`
- `assets/`
- `data/`
- `artifacts/`

Rule: put experiment materials, custom configs, generated data, and training outputs here.

## 3) Runtime Data Location (Canonical)

- `local_workspace/data/RawData/`
- `local_workspace/data/generated_datasets/`
- `local_workspace/artifacts/NIPS_results/`
- `local_workspace/artifacts/NIPS_results_test/`

## 4) Practical Guidance

- If you sync upstream, work in root source files first.
- If you run experiments, use configs/scripts under `local_workspace/`.
- Keep large outputs out of root and inside `local_workspace/artifacts/`.
