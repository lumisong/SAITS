# 数据生成脚本文件

data_downloading.sh
data_processing_utils.py
dataset_generating.sh
gene_ETTm1_dataset.py
gene_NRTSI_dataset.py
gene_PhysioNet2012_dataset.py
gene_UCI_BeijingAirQuality_dataset.py
gene_UCI_electricity_dataset.py
README.md

生成和预处理不同数据集

## 文件说明

1. data_downloading.sh - 用于下载原始数据集的脚本。 shell 脚本，用于自动执行数据集数据下载过程。下载的具体数据集取决于脚本的内容。
2. data_processing_utils.py - 预处理数据的通用工具代码，可能包含用于处理数据的实用函数。这些功能可能包括清理数据、标准化数据、处理缺失值、将数据分成训练集、验证集和测试集等的功能。
3. dataset_generating.sh - 生成数据集的主要脚本，可能自动生成数据集的 shell 脚本。
4. gene_XXX_dataset.py - 根据不同的数据集,生成对应数据集的脚本,包括:
   - ETTm1：ETT数据集
   - NRTSI：NRTSI数据集，实时流数据
   - PhysioNet2012：心电图数据
   - UCI_BeijingAirQuality: 北京空气质量数据
   - UCI_electricity: 电力消费数据
5. README.md - 目录说明文档

脚本可以用来自动下载不同公开数据集,并转换到该项目使用的统一格式,为模型的训练和评估提供数据。
