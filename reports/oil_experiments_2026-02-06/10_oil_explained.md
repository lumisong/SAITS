# 石油数据集实验说明（自动整理，2026-02-06）

你关心的数据集：`DaQing`、`NewZealand`、`NorthSea`（均为测井/石油相关数据）。

## 1) 指标是什么意思（人话）

- `best_imputation_mae_from_ckpt_name`：插补误差（MAE），越小越好。
- `best_ckpt_train_step`：最佳结果出现的训练步数。
- `checkpoint_count`：该实验保存了多少次模型。
- `total_size_gb`：该实验目录占用空间。
- `model_type`：模型类型（BRITS / MRNN / SAITS / Transformer）。

> 说明：MAE 来自 checkpoint 文件名解析，不是重新评测。

## 2) 你的石油实验总览（按数据集）

### DaQing（测井）

| 排名 | 模型 | 最佳MAE(越小越好) | 最佳训练步 | 保存模型数 | 占用空间(GB) | 实验目录 |
|---:|---|---:|---:|---:|---:|---|
| 1 | SAITS | 0.1095 | 189 | 23 | 0.117 | DaQing_SAITS_base |
| 2 | BRITS | 0.1104 | 196 | 30 | 0.018 | DaQing_BRITS_best |
| 3 | Transformer | 0.123 | 189 | 21 | 1.078 | DaQing_Transformer_best |
| 4 | SAITS | 0.1318 | 182 | 16 | 0.542 | DaQing_SAITS_best |
| 5 | MRNN | 0.5901 | 196 | 28 | 1.554 | DaQing_MRNN_best |

### NewZealand（测井）

| 排名 | 模型 | 最佳MAE(越小越好) | 最佳训练步 | 保存模型数 | 占用空间(GB) | 实验目录 |
|---:|---|---:|---:|---:|---:|---|
| 1 | BRITS | 0.0643 | 1169 | 78 | 0.659 | NewZeaLand_BRITS_best |
| 2 | SAITS | 0.0848 | 567 | 48 | 0.246 | NewZeaLand_SAITS_base |
| 3 | Transformer | 0.0948 | 644 | 31 | 1.594 | NewZeaLand_Transformer_best |
| 4 | SAITS | 0.1204 | 224 | 20 | 0.68 | NewZeaLand_SAITS_best |
| 5 | MRNN | 0.4471 | 217 | 27 | 1.509 | NewZeaLand_MRNN_best |

### NorthSea（测井）

| 排名 | 模型 | 最佳MAE(越小越好) | 最佳训练步 | 保存模型数 | 占用空间(GB) | 实验目录 |
|---:|---|---:|---:|---:|---:|---|
| 1 | BRITS | 0.0566 | 798 | 66 | 0.557 | NorthSea_BRITS_best |
| 2 | SAITS | 0.0771 | 812 | 51 | 0.261 | NorthSea_SAITS_base |
| 3 | SAITS | 0.1011 | 476 | 29 | 0.986 | NorthSea__SAITS_best |
| 4 | Transformer | 0.1148 | 287 | 24 | 1.234 | NorthSea_Transformer_best |
| 5 | MRNN | 0.4394 | 119 | 17 | 0.95 | NorthSea_MRNN_best |

## 3) 快速结论（只基于当前历史记录）

- DaQing（测井）：最好是 `SAITS`（MAE=0.1095），最差是 `MRNN`（MAE=0.5901）。
- NewZealand（测井）：最好是 `BRITS`（MAE=0.0643），最差是 `MRNN`（MAE=0.4471）。
- NorthSea（测井）：最好是 `BRITS`（MAE=0.0566），最差是 `MRNN`（MAE=0.4394）。

## 4) 你可以怎么用这份表

- 写汇报时：每个数据集只引用排名第1的模型 + MAE + checkpoint路径。
- 复现实验时：优先用 `run_name` 对应目录下的最佳 checkpoint。
- 不建议跨数据集直接比较 MAE 数值（数据难度不同）。

