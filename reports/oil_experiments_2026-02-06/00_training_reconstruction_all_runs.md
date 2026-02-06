# SAITS Training Work Reconstruction (Generated 2026-02-06)

This report is auto-reconstructed from existing artifacts under `SAITS/NIPS_results` and `SAITS/NIPS_results_test`.

## Overall

- Total run directories: 23
- `NIPS_results` runs: 19
- `NIPS_results_test` runs: 4
- Total artifact size: 28.592 GB
- Total checkpoint files: 996
- Total checkpoint size: 28.572 GB

## Largest Runs (Top 12 by Size)

| Scope | Run | Size (GB) | Checkpoints | Ckpt Size (GB) | Best MAE from ckpt name |
|---|---|---:|---:|---:|---:|
| NIPS_results | AirQuality_BRITS_best | 15.398 | 365 | 15.397 | 0.1423 |
| NIPS_results | NewZeaLand_Transformer_best | 1.594 | 31 | 1.593 | 0.0948 |
| NIPS_results | DaQing_MRNN_best | 1.554 | 28 | 1.554 | 0.5901 |
| NIPS_results | NewZeaLand_MRNN_best | 1.509 | 27 | 1.509 | 0.4471 |
| NIPS_results | NorthSea_Transformer_best | 1.234 | 24 | 1.234 | 0.1148 |
| NIPS_results | DaQing_Transformer_best | 1.078 | 21 | 1.078 | 0.123 |
| NIPS_results | NorthSea__SAITS_best | 0.986 | 29 | 0.986 | 0.1011 |
| NIPS_results | NorthSea_MRNN_best | 0.950 | 17 | 0.950 | 0.4394 |
| NIPS_results | NewZeaLand_SAITS_best | 0.680 | 20 | 0.680 | 0.1204 |
| NIPS_results | NewZeaLand_BRITS_best | 0.659 | 78 | 0.658 | 0.0643 |
| NIPS_results | NorthSea_BRITS_best | 0.557 | 66 | 0.557 | 0.0566 |
| NIPS_results | DaQing_SAITS_best | 0.542 | 16 | 0.542 | 0.1318 |

## Best MAE (from checkpoint filenames)

| Rank | Scope | Run | Model | Dataset | Best MAE | Best train step | Best val step |
|---:|---|---|---|---|---:|---:|---:|
| 1 | NIPS_results | NorthSea_BRITS_best | BRITS | NorthSea_seqlen100_02masked | 0.0566 | 798 | 114 |
| 2 | NIPS_results | NewZeaLand_BRITS_best | BRITS | NewZealand_seqlen100_02masked | 0.0643 | 1169 | 167 |
| 3 | NIPS_results | NorthSea_SAITS_base | SAITS | NorthSea_seqlen100_02masked | 0.0771 | 812 | 116 |
| 4 | NIPS_results | NewZeaLand_SAITS_base | SAITS | NewZealand_seqlen100_02masked | 0.0848 | 567 | 81 |
| 5 | NIPS_results | NewZeaLand_Transformer_best | Transformer | NewZealand_seqlen100_02masked | 0.0948 | 644 | 92 |
| 6 | NIPS_results | NorthSea__SAITS_best | SAITS | NorthSea_seqlen100_02masked | 0.1011 | 476 | 68 |
| 7 | NIPS_results | DaQing_SAITS_base | SAITS | DaQing_seqlen100_02masked | 0.1095 | 189 | 27 |
| 8 | NIPS_results | DaQing_BRITS_best | BRITS | DaQing_seqlen100_02masked | 0.1104 | 196 | 28 |
| 9 | NIPS_results | NorthSea_Transformer_best | Transformer | NorthSea_seqlen100_02masked | 0.1148 | 287 | 41 |
| 10 | NIPS_results | NewZeaLand_SAITS_best | SAITS | NewZealand_seqlen100_02masked | 0.1204 | 224 | 32 |
| 11 | NIPS_results | DaQing_Transformer_best | Transformer | DaQing_seqlen100_02masked | 0.1230 | 189 | 27 |
| 12 | NIPS_results | DaQing_SAITS_best | SAITS | DaQing_seqlen100_02masked | 0.1318 | 182 | 26 |
| 13 | NIPS_results | AirQuality_BRITS_best | BRITS | AirQuality_seqlen24_01masked | 0.1423 | 4907 | 701 |
| 14 | NIPS_results_test | AirQuality_SAITS_best | SAITS | AirQuality_seqlen24_01masked | 0.1608 | 343 | 49 |
| 15 | NIPS_results_test | AirQuality_SAITS_best_test | SAITS | AirQuality_seqlen24_01masked;AirQuality_seqlen24_01masked_test | 0.1608 | 343 | 49 |
| 16 | NIPS_results_test | AirQuality_SAITS_base | SAITS | AirQuality_seqlen24_01masked | 0.1693 | 350 | 50 |
| 17 | NIPS_results_test | AirQuality_MRNN_best | MRNN | AirQuality_seqlen24_01masked | 0.2818 | 105 | 15 |
| 18 | NIPS_results | NorthSea_MRNN_best | MRNN | NorthSea_seqlen100_02masked | 0.4394 | 119 | 17 |
| 19 | NIPS_results | NewZeaLand_MRNN_best | MRNN | NewZealand_seqlen100_02masked | 0.4471 | 217 | 31 |
| 20 | NIPS_results | DaQing_MRNN_best | MRNN | DaQing_seqlen100_02masked | 0.5901 | 196 | 28 |

## Per-Run Summary

| Scope | Run | Model | Dataset | Seq len | Features | Train set len | Params | Start | Last update | Size (GB) | Ckpts | Best MAE | Max train step | Config |
|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| NIPS_results | AirQuality_BRITS_best | BRITS | AirQuality_seqlen24_01masked | 24 | 132 | 850 | 11250848 | 2023-07-12 22:24:21 | 2023-07-14 13:56:14 | 15.398 | 365 | 0.1423 | 4907 | configs/AirQuality_BRITS_best.ini |
| NIPS_results | NewZeaLand_Transformer_best | Transformer | NewZealand_seqlen100_02masked | 100 | 15 | 38837 | 13692943 | 2023-07-19 13:26:37 | 2023-07-19 14:52:01 | 1.594 | 31 | 0.0948 | 644 | configs/NewZeaLand/NewZealand_Transformer_best.ini |
| NIPS_results | DaQing_MRNN_best | MRNN | DaQing_seqlen100_02masked | 100 | 10 | 246 | 10705326 | 2023-07-18 15:59:11 | 2023-07-18 16:23:49 | 1.554 | 28 | 0.5901 | 196 | configs\\DaQing\\DaQing_MRNN_best.ini |
| NIPS_results | NewZeaLand_MRNN_best | MRNN | NewZealand_seqlen100_02masked | 100 | 15 | 38837 | 10808241 | 2023-07-19 12:51:54 | 2023-07-19 13:26:33 | 1.509 | 27 | 0.4471 | 217 | configs/NewZeaLand/NewZealand_MRNN_best.ini |
| NIPS_results | NorthSea_Transformer_best | Transformer | NorthSea_seqlen100_02masked | 100 | 15 | 7418 | 13692943 | 2023-07-18 20:44:44 | 2023-07-18 20:59:39 | 1.234 | 24 | 0.1148 | 287 | .\\configs\\NorthSea\\NorthSea_Transformer_best.ini |
| NIPS_results | DaQing_Transformer_best | Transformer | DaQing_seqlen100_02masked | 100 | 10 | 246 | 13677578 | 2023-07-18 16:57:50 | 2023-07-18 17:34:56 | 1.078 | 21 | 0.123 | 189 | configs\\DaQing\\DaQing_Transformer_best.ini |
| NIPS_results | NorthSea__SAITS_best | SAITS | NorthSea_seqlen100_02masked | 100 | 15 | 7418 | 9019610 | 2023-07-19 09:44:44 | 2023-07-19 10:06:04 | 0.986 | 29 | 0.1011 | 476 | .\\configs\\NorthSea\\NorthSea_SAITS_best.ini |
| NIPS_results | NorthSea_MRNN_best | MRNN | NorthSea_seqlen100_02masked | 100 | 15 | 7418 | 10808241 | 2023-07-18 20:05:39 | 2023-07-18 20:25:23 | 0.950 | 17 | 0.4394 | 119 | .\\configs\\NorthSea\\NorthSea_MRNN_best.ini |
| NIPS_results | NewZeaLand_SAITS_best | SAITS | NewZealand_seqlen100_02masked | 100 | 15 | 38837 | 9019610 | 2023-07-19 15:53:58 | 2023-07-19 16:27:40 | 0.680 | 20 | 0.1204 | 224 | configs/NewZeaLand/NewZealand_SAITS_best.ini |
| NIPS_results | NewZeaLand_BRITS_best | BRITS | NewZealand_seqlen100_02masked | 100 | 15 | 38837 | 2261888 | 2023-07-19 10:19:16 | 2023-07-19 12:51:50 | 0.659 | 78 | 0.0643 | 1169 | configs/NewZeaLand/NewZealand_BRITS_best.ini |
| NIPS_results | NorthSea_BRITS_best | BRITS | NorthSea_seqlen100_02masked | 100 | 15 | 7418 | 2261888 | 2023-07-18 19:08:15 | 2023-07-18 19:44:00 | 0.557 | 66 | 0.0566 | 798 | .\\configs\\NorthSea\\NorthSea_BRITS_best.ini |
| NIPS_results | DaQing_SAITS_best | SAITS | DaQing_seqlen100_02masked | 100 | 10 | 246 | 8988120 | 2023-07-18 17:29:58 | 2023-07-18 18:40:51 | 0.542 | 16 | 0.1318 | 182 | configs\\DaQing\\DaQing_SAITS_best.ini |
| NIPS_results | NorthSea_SAITS_base | SAITS | NorthSea_seqlen100_02masked | 100 | 15 | 7418 | 1341914 | 2023-07-18 21:07:09 | 2023-07-18 21:35:59 | 0.261 | 51 | 0.0771 | 812 | .\\configs\\NorthSea\\NorthSea_SAITS_base.ini |
| NIPS_results | NewZeaLand_SAITS_base | SAITS | NewZealand_seqlen100_02masked | 100 | 15 | 38837 | 1341914 | 2023-07-19 14:52:04 | 2023-07-19 15:53:55 | 0.246 | 48 | 0.0848 | 581 | configs/NewZeaLand/NewZealand_SAITS_base.ini |
| NIPS_results | DaQing_SAITS_base | SAITS | DaQing_seqlen100_02masked | 100 | 10 | 246 | 1333464 | 2023-07-18 16:25:49 | 2023-07-18 16:49:26 | 0.117 | 23 | 0.1095 | 189 | configs\\DaQing\\DaQing_SAITS_base.ini |
| NIPS_results | Electricity_SAITS_best | SAITS | Electricity_seqlen100_01masked | 100 | 370 | 816 | 11511000 | 2023-07-19 09:43:13 | 2023-07-19 09:44:21 | 0.087 | 2 | 0.7111 | 14 | .\\configs\\NorthSea\\NorthSea_SAITS_best.ini |
| NIPS_results | DaQing_BRITS_best | BRITS | DaQing_seqlen100_02masked | 100 | 10 | 246 | 159856 | 2023-07-18 15:27:15 | 2023-07-18 15:55:37 | 0.018 | 30 | 0.1104 | 196 | configs\\DaQing\\DaQing_BRITS_best.ini |
| NIPS_results | AirQuality_SAITS_best_test | SAITS | - | - | - | - | - | 2023-07-18 14:13:01 | 2023-07-18 14:13:16 | 0.017 | 0 | - | - | - |
| NIPS_results | Electricity_Transformer_best | Transformer | DaQing_seqlen100_02masked | 100 | 10 | - | 13677578 | 2023-07-18 16:56:39 | 2023-07-18 16:56:39 | 0.000 | 0 | - | - | configs\\DaQing\\DaQing_Transformer_best.ini |
| NIPS_results_test | AirQuality_SAITS_best | SAITS | AirQuality_seqlen24_01masked | 24 | 132 | 850 | 3072656 | 2023-07-14 17:07:10 | 2023-07-14 17:25:26 | 0.403 | 35 | 0.1608 | 343 | .\\configs\\AirQuality_SAITS_best_test.ini |
| NIPS_results_test | AirQuality_SAITS_best_test | SAITS | AirQuality_seqlen24_01masked;AirQuality_seqlen24_01masked_test | 24 | 132 | 850 | 3072656 | 2023-07-18 10:04:19 | 2023-07-18 14:13:20 | 0.403 | 35 | 0.1608 | 343 | configs/AirQuality_SAITS_best_test_7_18.ini |
| NIPS_results_test | AirQuality_SAITS_base | SAITS | AirQuality_seqlen24_01masked | 24 | 132 | 850 | 1558160 | 2023-07-14 15:45:18 | 2023-07-14 16:03:49 | 0.205 | 35 | 0.1693 | 350 | .\\configs\\AirQuality_SAITS_base_test.ini |
| NIPS_results_test | AirQuality_MRNN_best | MRNN | AirQuality_seqlen24_01masked | 24 | 132 | 850 | 1404876 | 2023-07-14 15:04:14 | 2023-07-14 15:13:56 | 0.094 | 15 | 0.2818 | 105 | .\\configs\\AirQuality_MRNN_best_test.ini |

## Notes

- `best_imputation_mae_from_ckpt_name` comes from checkpoint filenames, not re-evaluated metrics.
- If a run has no parsed logs, model/dataset fields may be inferred from run name only.
- A machine-readable version is provided in `training_work_reconstruction_2026-02-06.csv`.
