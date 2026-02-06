# Oil Experiments Summary

Date: 2026-02-06
Scope: `DaQing`, `NewZealand`, `NorthSea`

## How To Read

- `best_mae`: best imputation MAE from checkpoint names, lower is better.
- `duration_hours`: approximate run duration from log start/end times.
- `param_estimated_mb`: parameter size estimate (`trainable_params * 4 bytes`).
- `experiment_size_gb`: total disk usage of this run directory.

## DaQing

- Best MAE: `SAITS` (0.1095)
- Fastest run: `SAITS` (0.39 h)
- Smallest model: `BRITS` (0.61 MB)

| Rank | Model | Best MAE | Duration (h) | Param (MB) | Run Size (GB) | CKPT Count | Run Name |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | SAITS | 0.1095 | 0.39 | 5.09 | 0.117 | 23 | DaQing_SAITS_base |
| 2 | BRITS | 0.1104 | 0.47 | 0.61 | 0.018 | 30 | DaQing_BRITS_best |
| 3 | Transformer | 0.1230 | 0.62 | 52.18 | 1.078 | 21 | DaQing_Transformer_best |
| 4 | SAITS | 0.1318 | 1.18 | 34.29 | 0.542 | 16 | DaQing_SAITS_best |
| 5 | MRNN | 0.5901 | 0.41 | 40.84 | 1.554 | 28 | DaQing_MRNN_best |

## NewZealand

- Best MAE: `BRITS` (0.0643)
- Fastest run: `SAITS` (0.56 h)
- Smallest model: `SAITS` (5.12 MB)

| Rank | Model | Best MAE | Duration (h) | Param (MB) | Run Size (GB) | CKPT Count | Run Name |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | BRITS | 0.0643 | 2.54 | 8.63 | 0.659 | 78 | NewZeaLand_BRITS_best |
| 2 | SAITS | 0.0848 | 1.03 | 5.12 | 0.246 | 48 | NewZeaLand_SAITS_base |
| 3 | Transformer | 0.0948 | 1.42 | 52.23 | 1.594 | 31 | NewZeaLand_Transformer_best |
| 4 | SAITS | 0.1204 | 0.56 | 34.41 | 0.680 | 20 | NewZeaLand_SAITS_best |
| 5 | MRNN | 0.4471 | 0.58 | 41.23 | 1.509 | 27 | NewZeaLand_MRNN_best |

## NorthSea

- Best MAE: `BRITS` (0.0566)
- Fastest run: `Transformer` (0.25 h)
- Smallest model: `SAITS` (5.12 MB)

| Rank | Model | Best MAE | Duration (h) | Param (MB) | Run Size (GB) | CKPT Count | Run Name |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | BRITS | 0.0566 | 0.60 | 8.63 | 0.557 | 66 | NorthSea_BRITS_best |
| 2 | SAITS | 0.0771 | 0.48 | 5.12 | 0.261 | 51 | NorthSea_SAITS_base |
| 3 | SAITS | 0.1011 | 0.36 | 34.41 | 0.986 | 29 | NorthSea__SAITS_best |
| 4 | Transformer | 0.1148 | 0.25 | 52.23 | 1.234 | 24 | NorthSea_Transformer_best |
| 5 | MRNN | 0.4394 | 0.33 | 41.23 | 0.950 | 17 | NorthSea_MRNN_best |

