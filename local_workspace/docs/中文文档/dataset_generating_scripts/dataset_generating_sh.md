# 数据生成自动化脚本

用于运行数据集生成 python 脚本的 shell 脚本。 dataset_generating.sh 脚本是生成 SAITS 模型使用的数据集的有用工具

## 文件说明

1. 从注释中可以看出,在使用代码时需要引用相关论文。
2. 为PhysioNet 2012数据集生成physio2012_37feats_01masked_1数据集。
3. 为UCI北京空气质量数据集生成AirQuality_seqlen24_01masked数据集。
4. 为UCI电力消费数据集生成Electricity_seqlen100_01masked数据集。
5. 为ETTm1数据集生成ETTm1_seqlen24_01masked数据集。

这个脚本会针对原始的数据集做序列长度的提取、丢失率的设置等处理,最终生成适合于时间序列插补任务的数据集,保存在generated_datasets目录下。

## 表格

| 数据集名称 | 数据集生成 | 数据集长度序列 | 数据集丢失率 | 数据集描述 |
| :---: | :---: | :---: | :---: | :---: |
| PhysioNet 2012 | physio2012_37feats_01masked_1 | 24 | 0.1 | 37个特征,序列长度24,丢失率0.1 |
| 空气质量 | AirQuality_seqlen24_01masked | 24 | 0.1 | 14个特征,序列长度24,丢失率0.1 |
| 电力消费 | Electricity_seqlen100_01masked | 100 | 0.1 | 321个特征,序列长度100,丢失率0.1 |
| ETTm1 | ETTm1_seqlen24_01masked | 24 | 0.1 | 321个特征,序列长度24,丢失率0.1 |

注意：

1. The path to the raw dataset.原始数据集的路径。
2. The length of the sequences.序列的长度。
3. The rate of artificial missing values.人为缺失值的比率。
4. The sliding window length.滑动窗口的长度。

## 存储位置

1. gene_PhysioNet2012_dataset.py：此脚本使用 RawData/Physio2012_mega/mega 中存储的原始数据生成 PhysioNet-2012 数据集。它使用存储在 RawData/Physio2012_mega/ 的结果文件。处理后的数据集以 physio2012_37feats_01masked_1 形式保存在 ../generated_datasets 目录中。
2. gene_UCI_BeijingAirQuality_dataset.py：此脚本使用 RawData/AirQuality/PRSA_Data_20130301-20170228 中存储的原始数据生成 UCI 北京空气质量数据集。设定序列长度为24，人工缺失率为0.1。处理后的数据集以 AirQuality_seqlen24_01masked 形式保存在 ../generated_datasets 目录中。
3. gene_UCI_electricity_dataset.py：此脚本使用 RawData/Electricity/LD2011_2014.txt 中存储的原始数据生成 UCI 电力数据集。设定序列长度为100，人工缺失率为0.1。处理后的数据集以 Electricity_seqlen100_01masked 形式保存在 ../generated_datasets 目录中。
4. gene_ETTm1_dataset.py：此脚本使用 RawData/ETT/ETTm1.csv 中存储的原始数据生成电力变压器温度 (ETT) 数据集。设定序列长度为24，滑动长度为12，人工缺失率为0.1。处理后的数据集以 ETTm1_seqlen24_01masked 形式保存在 ../generated_datasets 目录中。

要运行此脚本，您需要在计算机上安装 Python 并安装每个 Python 脚本所需的依赖项。这些脚本通常需要 Python 数据处理库（例如 pandas），也可能需要其他库。

确保原始数据位于脚本中指定的正确目录中。
