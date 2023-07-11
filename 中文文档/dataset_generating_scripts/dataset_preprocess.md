# 数据集准备过程细节

本部分涉及到：五个数据集，下面将依次来讲解这部分数据集

## ETTm1

### 数据集介绍

略

### 处理过程

gene_ETTm1_dataset.py

#### 源代码_ETTm1

```python
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ETTm1 dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument("--sliding_len", help="sequence length", type=int, default=12)
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate ETTm1 dataset",
        mode="w",
    )
    logger.info(args)

    df = pd.read_csv(args.file_path, index_col="date")
    df.index = pd.to_datetime(df.index)
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df["datetime"] = pd.to_datetime(df.index)

    unique_months = df["datetime"].dt.to_period("M").unique()
    selected_as_test = unique_months[:4]  # select first 4 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[4:8]  # select the 4th - the 8th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[8:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len, args.sliding_len)
    val_set_X = window_truncate(val_set_X, args.seq_len, args.sliding_len)
    test_set_X = window_truncate(test_set_X, args.seq_len, args.sliding_len)

    # add missing values in train set manually
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)
        logger.info(
            f"Already masked out {args.artificial_missing_rate * 100}% values in train set"
        )

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }
    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])
    total_sample_num = train_sample_num + val_sample_num + test_sample_num
    logger.info(
        f"Feature num: {feature_num},\n"
        f"{train_sample_num} ({(train_sample_num / total_sample_num):.3f}) samples in train set\n"
        f"{val_sample_num} ({(val_sample_num / total_sample_num):.3f}) samples in val set\n"
        f"{test_sample_num} ({(test_sample_num / total_sample_num):.3f}) samples in test set\n"
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, "scaler"))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
```

#### 代码说明_ETTm1数据集生成脚本

生成ETTm1数据集的脚本,主要流程包括:

1. 从命令行参数中读取配置,包括文件路径、丢失率、序列长度等。
2. 载入原始CSV数据,提取特征和时间信息。
3. 根据时间将数据划分为训练集、验证集和测试集。
4. 对数据进行标准化处理。
5. 用window_truncate函数截取固定长度的序列样本。
6. 在训练集上人工添加丢失。
7. 用add_artificial_mask函数在验证集和测试集上添加丢失。
8. 统计并打印数据集信息。
9. 调用saving_into_h5函数将划分后的数据集保存到h5文件。
10. 保存标准化器以备测试时反标准化。
11. 记录日志信息。

总体上利用了data_processing_utils.py中的通用函数,实现了根据ETTm1源数据生成时间序列样例数据的全流程,包括划分、预处理、打散样本等操作。

- file_path ：ETTm1 数据集文件的路径。
- artificial_missing_rate ：将要创建的人为缺失值的比率。
- seq_len ：序列的长度。
- sliding_len ：滑动窗口的大小。
- dataset_name ：生成的数据集的名称。
- saving_path ：生成的数据集的父目录。

该脚本首先读取 ETTm1 数据集文件并将其分为三组：train、val 和 test。

训练集用于训练SAITS模型，验证集用于评估SAITS模型的性能，测试集用于测试SAITS模型的性能。

该脚本预处理这三组数据。预处理步骤包括：

- 将数据缩放至均值和单位方差为零。
- 将数据截断为给定长度的序列.
- 向数据添加人工缺失值。

预处理后的数据保存在HDF5文件中。然后，HDF5 文件可用于训练和评估 SAITS 模型。

用于预处理 ETTm1 数据集以执行时间序列建模任务，例如预测。

以下是该脚本功能的简要说明：

1. 首先设置一个参数解析器来获取数据集的文件路径、人工缺失率、序列长度、滑动长度、数据集名称以及生成的数据集的保存路径。
2. 使用提供的保存路径和数据集名称创建一个新目录来保存数据集。
3. 设置记录器来跟踪脚本的进程，并将日志消息保存到名为“dataset_generate.log”的文件中。
4. 脚本使用 pandas 将 ETTm1 数据集文件读取到 DataFrame 中，并处理日期列以确保其采用日期时间格式。它还从 DataFrame 中提取特征名称和特征数量。
5. 根据数据的月份将数据集分为训练集、验证集和测试集。前4个月作为测试集，接下来的4个月作为验证集，其余作为训练集。
6. 使用 sklearn 的 StandardScaler 缩放训练、验证和测试集中的特征。这确保了特征的平均值为 0，标准差为 1，这有助于改进模型训练。
7. 将 window_truncate 函数应用于每个集合以生成时间序列样本。
8. 如果人工缺失率大于0，则脚本通过调用 random_mask 函数将人工缺失值引入训练集中。
9. 使用 add_artificial_mask 函数向每个集合添加缺失值和掩码。
10. 记录验证和测试集中人工屏蔽值的数量。
11. 处理后的数据被收集到字典中，并计算每组中的样本数量。
12. 最终使用 saving_into_h5 函数将处理后的数据保存到 HDF5 文件中，并将 StandardScaler 对象保存为 pickle 文件以供以后在数据转换中使用。
13. 该脚本会记录一条消息，以指示该过程已完成以及处理后的数据已保存在何处。

## NRTSI数据集

### 数据集介绍

NRTSI数据集是由[1]提出的用于时间序列插值的数据集，包含了多个真实世界的时间序列数据，包括气象、交通、能源等领域的数据。NRTSI数据集中的时间序列数据包含了缺失值，用于时间序列插值任务的训练和测试。

### 数据集下载

NRTSI数据集可以从[这里]()

### 数据集生成

#### 源代码_NRSTI数据集

```python
import argparse
import os
import sys

import numpy as np

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets from NRTSI")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate NRTSI dataset",
        mode="w",
    )
    logger.info(args)

    train_set_X = np.load(os.path.join(args.file_path, "train.npy"))
    val_set_X = np.load(os.path.join(args.file_path, "val.npy"))
    test_set_X = np.load(os.path.join(args.file_path, "test.npy"))

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )

    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }

    args.feature_num = train_set_X.shape[-1]
    logger.info(
        f"Feature num: {args.feature_num},\n"
        f'Sample num in train set: {len(train_set_dict["X"])}\n'
        f'Sample num in val set: {len(val_set_dict["X"])}\n'
        f'Sample num in test set: {len(test_set_dict["X"])}\n'
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
```

#### 代码解读_NRSTI数据集

gene_NRTSI_dataset.py 似乎与前面的脚本 gene_ETTm1_dataset.py 非常相似，只是名称不同。这两个脚本都用于预处理给定的时间序列数据集，应用缩放等操作，引入人工缺失值以及将数据集拆分为训练集、验证集和测试集。

脚本功能的简要概述：

1. 解析指定详细信息的命令行参数，例如数据集文件路径、要引入的人工缺失值的比例、时间序列样本的序列长度、创建重叠样本的滑动长度、生成的数据集的名称和目录将其保存在其中。
2. 检查输出目录是否存在，如果不存在，则创建它。
3. 设置记录器来跟踪进程并将日志消息保存到名为“dataset_generate.log”的文件中。
4. 将指定的数据集文件读入 pandas DataFrame，并提取特征名称和计数。假定索引包含日期，这些日期将转换为日期时间格式。
5. 根据日期将数据集分为训练集、验证集和测试集。前4个月作为测试集，接下来的4个月作为验证集，其余作为训练集。
6. 使用 sklearn 中的 StandardScaler 缩放数据集中的特征。这将特征标准化为具有零均值和单位方差，这是训练许多机器学习模型之前的常见做法。
7. 使用 window_truncate 函数从每组生成时间序列样本。
8. 如果指定大于零的人工缺失率，则会将该比例的缺失值引入训练集中。
9. 使用 add_artificial_mask 函数向每个集合添加人工缺失值和掩码。
10. 记录验证和测试集中人工屏蔽值的数量。
11. 将处理后的数据收集到字典中，计算每组中的样本数量，并记录这些数字。
12. 使用 saving_into_h5 函数将处理后的数据保存到HDF5文件中。它还将 StandardScaler 保存为 pickle 文件以供将来使用。
13. 记录一条消息，指示该过程已完成以及已处理数据的保存位置。

虽然这些脚本似乎是针对不同的数据集（ETTm1 和 NRTSI）命名的，但它们执行的操作是相同的，并且它们可用于预处理适合假设结构的任何时间序列数据集（列中的特征、时间按日期索引）。

他们实际处理的数据集将取决于运行脚本时提供的文件路径。

gene_NRTSI_dataset.py 用于生成NRTSI数据集

- file_path ：NRTSI 数据集文件的路径。
- artificial_missing_rate ：将要创建的人为缺失值的比率。
- seq_len ：序列的长度。
- sliding_len ：滑动窗口的大小。
- dataset_name ：生成的数据集的名称。
- saving_path ：生成的数据集的父目录。

首先读取 NRTSI 数据集文件并将其分为三组：train、val 和 test。

训练集用于训练SAITS模型，验证集用于评估SAITS模型的性能，测试集用于测试SAITS模型的性能。

脚本预处理这三组数据。预处理步骤包括：

- 将数据缩放至均值和单位方差为零。
- 将数据截断为给定长度的序列。
- 向数据添加人工缺失值。

预处理后的数据保存在HDF5文件中。然后，HDF5 文件可用于训练和评估 SAITS 模型。

脚本的主要逻辑与ETTm1的数据集生成脚本基本一致,区别在于:

1. 数据集来源改为NRSTI的数据文件。
2. 描述信息将数据集名改为NRSTI。
3. 其余处理流程,包括读取数据、划分数据集、标准化、窗口截取、添加丢失、保存为h5文件等完全相同。

这样就可以通过统一的生成流程,分别处理ETTm1和NRSTI两个时间序列数据集,产生规范化的训练、验证、测试数据集。

同时避免了重复代码,利用了data_processing_utils.py中定义的通用时间序列处理函数,提高了代码的复用性。

只需要修改源数据文件,就可以快速生成不同数据集的示例数据,十分方便。
这种脚本化的处理方式,使得时间序列数据的预处理流程标准化,可以批量生产所需的样本数据。 

## PhysioNet2012 数据集

gene_PhysioNet2012_dataset.py

### 数据集生成_physioNet2012

略

#### 源代码

```python

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    add_artificial_mask,
    saving_into_h5,
)

np.random.seed(26)


def process_each_set(set_df, all_labels):
    # gene labels, y
    sample_ids = set_df["RecordID"].to_numpy().reshape(-1, 48)[:, 0]
    y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)
    # gene feature vectors, X
    set_df = set_df.drop("RecordID", axis=1)
    feature_names = set_df.columns.tolist()
    X = set_df.to_numpy()
    X = X.reshape(len(sample_ids), 48, len(feature_names))
    return X, y, feature_names


def keep_only_features_to_normalize(all_feats, to_remove):
    for i in to_remove:
        all_feats.remove(i)
    return all_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PhysioNet2012 dataset")
    parser.add_argument(
        "--raw_data_path", help="path of physio 2012 raw dataset", type=str
    )
    parser.add_argument(
        "--outcome_files_dir", help="dir path of raw dataset's outcome file", type=str
    )
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--train_frac", help="fraction of train set", type=float, default=0.8
    )
    parser.add_argument(
        "--val_frac", help="fraction of validation set", type=float, default=0.2
    )
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    # create saving dir
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    # set up logger
    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate PhysioNet2012 dataset",
        mode="w",
    )
    logger.info(args)

    outcome_files = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    outcome_collector = []
    for o_ in outcome_files:
        outcome_file_path = os.path.join(args.outcome_files_dir, o_)
        with open(outcome_file_path, "r") as f:
            outcome = pd.read_csv(f)[["In-hospital_death", "RecordID"]]
        outcome = outcome.set_index("RecordID")
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []
    for filename in os.listdir(args.raw_data_path):
        recordID = int(filename.split(".txt")[0])
        with open(os.path.join(args.raw_data_path, filename), "r") as f:
            df_temp = pd.read_csv(f)
        df_temp["Time"] = df_temp["Time"].apply(lambda x: int(x.split(":")[0]))
        df_temp = df_temp.pivot_table("Value", "Time", "Parameter")
        df_temp = df_temp.reset_index()  # take Time from index as a col
        if len(df_temp) == 1:
            logger.info(
                f"Pass {recordID}, because its len==1, having no time series data"
            )
            continue
        all_recordID.append(recordID)  # only count valid recordID
        if df_temp.shape[0] != 48:
            missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
            missing_part = pd.DataFrame({"Time": missing})
            df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)
            df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[
            :48
        ]  # only take 48 hours, some samples may have more records, like 49 hours
        df_temp["RecordID"] = recordID
        df_temp["Age"] = df_temp.loc[0, "Age"]
        df_temp["Height"] = df_temp.loc[0, "Height"]
        df_collector.append(df_temp)
    df = pd.concat(df_collector, sort=True)
    df = df.drop(["Age", "Gender", "ICUType", "Height"], axis=1)
    df = df.reset_index(drop=True)
    df = df.drop("Time", axis=1)  # dont need Time col

    train_set_ids, test_set_ids = train_test_split(
        all_recordID, train_size=args.train_frac
    )
    train_set_ids, val_set_ids = train_test_split(
        train_set_ids, test_size=args.val_frac
    )
    logger.info(f"There are total {len(train_set_ids)} patients in train set.")
    logger.info(f"There are total {len(val_set_ids)} patients in val set.")
    logger.info(f"There are total {len(test_set_ids)} patients in test set.")

    all_features = df.columns.tolist()
    feat_no_need_to_norm = ["RecordID"]
    feats_to_normalize = keep_only_features_to_normalize(
        all_features, feat_no_need_to_norm
    )

    train_set = df[df["RecordID"].isin(train_set_ids)]
    val_set = df[df["RecordID"].isin(val_set_ids)]
    test_set = df[df["RecordID"].isin(test_set_ids)]

    # standardization
    scaler = StandardScaler()
    train_set.loc[:, feats_to_normalize] = scaler.fit_transform(
        train_set.loc[:, feats_to_normalize]
    )
    val_set.loc[:, feats_to_normalize] = scaler.transform(
        val_set.loc[:, feats_to_normalize]
    )
    test_set.loc[:, feats_to_normalize] = scaler.transform(
        test_set.loc[:, feats_to_normalize]
    )

    train_set_X, train_set_y, feature_names = process_each_set(train_set, all_outcomes)
    val_set_X, val_set_y, _ = process_each_set(val_set, all_outcomes)
    test_set_X, test_set_y, _ = process_each_set(test_set, all_outcomes)

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    train_set_dict["labels"] = train_set_y
    val_set_dict["labels"] = val_set_y
    test_set_dict["labels"] = test_set_y

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }

    logger.info(f"All saved features: {feature_names}")
    saved_df = df.loc[:, feature_names]

    total_sample_num = 0
    total_positive_num = 0
    for set_name, rec in zip(
        ["train", "val", "test"], [train_set_dict, val_set_dict, test_set_dict]
    ):
        total_sample_num += len(rec["labels"])
        total_positive_num += rec["labels"].sum()
        logger.info(
            f'Positive rate in {set_name} set: {rec["labels"].sum()}/{len(rec["labels"])}='
            f'{(rec["labels"].sum() / len(rec["labels"])):.3f}'
        )
    logger.info(
        f"Dataset overall positive rate: {(total_positive_num / total_sample_num):.3f}"
    )

    missing_part = np.isnan(saved_df.to_numpy())
    logger.info(
        f"Dataset overall missing rate of original feature vectors (without any artificial mask): "
        f"{(missing_part.sum() / missing_part.shape[0] / missing_part.shape[1]):.3f}"
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")
```

#### 代码解析_physionet2012.py

脚本“gene_PhysioNet2012_dataset.py”处理原始 PhysioNet 2012 数据集，并将其转换为适合机器学习任务的格式，特别是涉及时间序列数据的任务。它涉及多个预处理步骤，包括数据加载、清理、标准化以及分割为训练、验证和测试集。以下是其主要部分的细分：

1. **导入所需的库**：脚本首先从“sklearn.preprocessing”导入所有必需的Python库，例如“numpy”、“pandas”、“argparse”、“os”、“sys”、“StandardScaler” ， 和更多。

2. **函数定义**：接下来，脚本定义了几个函数：
   -`process_each_set()`：该函数从每个集合中提取特征向量和标签（训练/验证/测试）。
-`keep_only_features_to_normalize()`：此函数从所有特征列表中删除不应标准化的特征。

3. **主脚本（if __name__ == "__main__"）**：这是脚本主要执行开始的地方。它执行以下操作：
   -它设置参数解析以允许各种设置的命令行参数，例如原始数据路径、用于训练/验证的数据部分等。
-它设置一个记录器来记录执行期间的事件。
   -它加载并处理原始数据，包括旋转和清理。
   -它将数据分为训练集、验证集和测试集。
   -它标准化了特征。
   -它对数据应用人工掩码。
   -它计算并记录各种统计数据，例如阳性率和缺失率。
-它将处理后的数据以及经过训练的缩放器对象分别保存到“.h5”文件和 pickle 文件中。

Python 脚本 gene_PhysioNet2012_dataset.py 是从 PhysioNet 2012 数据集生成数据集的脚本。该脚本采用以下参数：

- `raw_data_path` ：PhysioNet 2012 原始数据集的路径。
- outcome_files_dir ：原始数据集结果文件的目录路径。
- train_frac ：训练集的分数。
- val_frac ：验证集的分数。
- dataset_name ：生成的数据集的名称。
- saving_path ：生成的数据集的父目录。

脚本首先读取 PhysioNet 2012 原始数据集并将其分为三组：训练集、验证集和测试集。

预处理这三组数据。预处理步骤包括：

- 标准化：数据经过标准化，均值和单位方差为零。
- 添加人工缺失值：将人工缺失值添加到数据中以模拟真实世界的数据。
- 将数据拆分为序列：将数据拆分为给定长度的序列。

预处理后的数据保存在HDF5文件中。然后，HDF5 文件可用于训练和评估 SAITS 模型。

生成PhysioNet 2012数据集的脚本,主要流程包括:

1. 从raw data目录读取源数据文件,整合到pandas DataFrame中。
2. 读取outcome文件,获取标签。
3. 划分训练集、验证集和测试集。
4. 标准化特征。
5. 将时间序列样本reshape为模型输入格式。
6. 在训练集、验证集、测试集上添加人工丢失。
7. 保存标签。
8. 统计并打印数据集信息,如正样本率、总体丢失率等。
9. 调用saving_into_h5函数保存到h5文件。
10. 保存标准化器。
11. 记录日志。

相比于时间序列预测任务,这里针对分类任务做了一些改动:

- 不需要窗口截取,直接用48小时序列作为样本
- 增加了标签处理、统计和保存
- 打印并记录了更多的元信息

利用通用功能函数使代码模块化,通过配置可以生成不同的生理时间序列数据集。

## UCI北京空气质量数据集

### 数据集生成_uci_beijing_air_quality.py

#### 源代码_uci_beijing_air_quality.py

```python
import argparse
import os
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UCI air quality dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=100)
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate UCI air quality dataset",
        mode="w",
    )
    logger.info(args)

    df_collector = []
    station_name_collector = []
    file_list = os.listdir(args.file_path)
    for filename in file_list:
        file_path = os.path.join(args.file_path, filename)
        current_df = pd.read_csv(file_path)
        current_df["date_time"] = pd.to_datetime(
            current_df[["year", "month", "day", "hour"]]
        )
        station_name_collector.append(current_df.loc[0, "station"])
        # remove duplicated date info and wind direction, which is a categorical col
        current_df = current_df.drop(
            ["year", "month", "day", "hour", "wd", "No", "station"], axis=1
        )
        df_collector.append(current_df)
        logger.info(f"reading {file_path}, data shape {current_df.shape}")

    logger.info(
        f"There are total {len(station_name_collector)} stations, they are {station_name_collector}"
    )
    date_time = df_collector[0]["date_time"]
    df_collector = [i.drop("date_time", axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    args.feature_names = [
        station + "_" + feature
        for station in station_name_collector
        for feature in df_collector[0].columns
    ]
    args.feature_num = len(args.feature_names)
    df.columns = args.feature_names
    logger.info(
        f"Original df missing rate: "
        f"{(df[args.feature_names].isna().sum().sum() / (df.shape[0] * args.feature_num)):.3f}"
    )

    df["date_time"] = date_time
    unique_months = df["date_time"].dt.to_period("M").unique()
    selected_as_test = unique_months[:10]  # select first 3 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[10:20]  # select the 4th - the 6th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[20:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["date_time"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date_time"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date_time"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, args.feature_names])
    val_set_X = scaler.transform(val_set.loc[:, args.feature_names])
    test_set_X = scaler.transform(test_set.loc[:, args.feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }

    logger.info(
        f"Feature num: {args.feature_num},\n"
        f'Sample num in train set: {len(train_set_dict["X"])}\n'
        f'Sample num in val set: {len(val_set_dict["X"])}\n'
        f'Sample num in test set: {len(test_set_dict["X"])}\n'
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")


```

#### 代码解析_uci_beijing_air_quality.py

该脚本基于空气质量数据生成数据集，假设以 CSV 格式存储。以下是该脚本功能的概述：

1. 它读取指定各种参数的命令行参数，例如 CSV 文件的路径、生成的数据集的名称、保存数据集的路径等。

2. 脚本准备一个日志文件来记录数据集生成的过程。
3. 然后，脚本从参数中指定的目录中读取每个 CSV 文件。对于每个 CSV，它都会执行一些预处理操作，例如将年、月、日和小时转换为日期时间列，并删除不必要的列。 CSV 文件中的所有数据都连接到一个 DataFrame 中。

4. 之后，脚本检查并记录 DataFrame 中的原始丢失数据率。
5. 然后，脚本根据 date_time 字段的月份将数据划分为训练集、验证集和测试集。前10个月用作测试集，接下来的10个月用作验证集，其余几个月用作训练集。

6. 然后使用仅适合训练集的“StandardScaler”对所有三组进行缩放。
7. 然后，它对数据应用函数“window_truncate”，大概是为了创建指定长度的序列以进行时间序列分析。

8. 此后，使用函数“add_artificial_mask”将人工掩码添加到数据中。此函数可能会向数据引入人工缺失值，以模拟传感器数据可能缺失条目的真实场景。
9. 最后，脚本使用函数“ saving_into_h5”将处理后的数据集保存到 HDF5 文件中，并存储拟合的“StandardScaler”实例以供将来使用。

总之，该脚本正在根据空气质量数据的原始 CSV 文件准备时间序列数据集，并添加人工缺失值来模拟真实场景。它将数据分为训练集、验证集和测试集，对其进行缩放，将其截断为序列，并保存生成的处理数据集。它还记录整个过程。该脚本将用作数据预处理管道的一部分，特别是用于与时间序列中缺失数据插补相关的任务。

Python 脚本 gene_UCI_BeijingAirQuality_dataset.py 是一个从 UCI 北京空气质量数据集生成数据集的脚本。该脚本采用以下参数：

- file_path ：UCI 北京空气质量数据集的路径。
- artificial_missing_rate ：将要创建的人为缺失值的比率。
- seq_len ：序列的长度。
- dataset_name ：生成的数据集的名称。
- saving_path ：生成的数据集的父目录。

首先读取 UCI 北京空气质量数据集，并将其分为三组：训练集、验证集和测试集。

训练集用于训练SAITS模型，验证集用于评估SAITS模型的性能，测试集用于测试SAITS模型的性能。

脚本预处理这三组数据。预处理步骤包括：

1. 标准化：数据经过标准化，均值和单位方差为零。
2. Windowing: 数据被加窗为给定长度的序列。
3. 添加人工缺失值：将人工缺失值添加到数据中以模拟真实世界的数据。

生成UCI北京空气质量数据集的脚本,主要逻辑是:

1. 读取多个站点的数据文件,合并到一个DataFrame中。
2. 设置datetime列并根据月份划分训练集、验证集和测试集。
3. 对特征进行标准化。
4. 进行窗口截取来生成样本。
5. 在训练集、验证集、测试集上人为引入缺失。
6. 保存为h5格式。
7. 记录日志。

这与生成时间序列预测数据集的主要逻辑基本一致。
与其他脚本不同的地方在于:

1. 这个数据集来自多个站点,需要先合并。
2. 数据没有标签,只进行时间序列预测。
3. 打印并保存了特征名称。
4. 计算并打印了原始数据的缺失率。
通过参数配置,该脚本可以轻松生成不同长度序列、缺失率、划分方案的空气质量时间序列数据集。

## UCI电力需求数据集

### 数据集生成_uci_electricity_demand.py

#### 源代码_uci_electricity_demand.py

```python
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsdb import pickle_dump

sys.path.append("..")
from modeling.utils import setup_logger
from dataset_generating_scripts.data_processing_utils import (
    window_truncate,
    random_mask,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UCI electricity dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=100)
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="."
    )
    args = parser.parse_args()

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    logger = setup_logger(
        os.path.join(dataset_saving_dir + "/dataset_generating.log"),
        "Generate UCI electricity dataset",
        mode="w",
    )
    logger.info(args)

    df = pd.read_csv(args.file_path, index_col=0, sep=";", decimal=",")
    df.index = pd.to_datetime(df.index)
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df["datetime"] = pd.to_datetime(df.index)

    unique_months = df["datetime"].dt.to_period("M").unique()
    selected_as_test = unique_months[:10]  # select first 10 months as test set
    logger.info(f"months selected as test set are {selected_as_test}")
    selected_as_val = unique_months[
        10:20
    ]  # select the 11th - the 20th months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_train = unique_months[20:]  # use left months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["datetime"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    # add missing values in train set manually
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)
        logger.info(
            f"Already masked out {args.artificial_missing_rate * 100}% values in train set"
        )

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    logger.info(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    logger.info(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
    }
    train_sample_num = len(train_set_dict["X"])
    val_sample_num = len(val_set_dict["X"])
    test_sample_num = len(test_set_dict["X"])
    total_sample_num = train_sample_num + val_sample_num + test_sample_num
    logger.info(
        f"Feature num: {feature_num},\n"
        f"{train_sample_num} ({(train_sample_num / total_sample_num):.3f}) samples in train set\n"
        f"{val_sample_num} ({(val_sample_num / total_sample_num):.3f}) samples in val set\n"
        f"{test_sample_num} ({(test_sample_num / total_sample_num):.3f}) samples in test set\n"
    )

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    logger.info(f"All done. Saved to {dataset_saving_dir}.")


```

#### 代码解析_uci_electricity_demand.py

该脚本用于根据 UCI 机器学习存储库中的用电量数据生成数据集。该过程与您询问的上一个脚本中的过程很大程度上相似，但有一些关键差异需要注意：

1. 此处仅读取一个 CSV 文件，而不是空气质量数据集脚本中的多个 CSV 文件。
2. 原始 CSV 文件的格式似乎有所不同，数据以分号分隔，并使用逗号作为小数点，这在某些国家/地区很常见。
3. 读取和预处理数据后，脚本再次根据日期时间字段的月份将其划分为训练集、验证集和测试集。该脚本还使用前 10 个月作为测试集，第 11 到 20 个月作为验证集，其余月份作为训练集。
4. 与空气质量脚本一样，每组中的数据都使用 StandardScaler 进行缩放。
5. 函数 window_truncate 应用于数据以创建指定长度的序列以进行时间序列分析。
6. 调用 random_mask 函数仅屏蔽掉训练集中的部分数据。要屏蔽的数据比例由 artificial_missing_rate 参数确定。请注意，在电力数据集脚本中，直接调用 random_mask 将 NaN 引入数据中，而在空气质量脚本中，使用 add_artificial_mask 函数来执行此任务。
7. 然后应用 add_artificial_mask 函数在训练、验证和测试集中引入人工缺失数据。
8. 使用函数 saving_into_h5 将处理后的数据集保存到 HDF5 文件中，并且还存储拟合的 StandardScaler 实例以供将来使用。

该脚本正在从用电数据的原始 CSV 文件准备时间序列数据集。它将数据分为训练集、验证集和测试集，对其进行缩放，将其截断为序列，引入人工缺失值，并保存生成的处理数据集。它还记录整个过程。与之前的脚本一样，它将用作数据预处理管道的一部分，特别是对于与时间序列中缺失数据插补相关的任务。

- 该脚本使用 random_mask 函数向数据添加人工缺失值。 random_mask 函数随机选择一部分数据点并将其设置为缺失。
- 该脚本使用 add_artificial_mask 函数向数据添加人工缺失值，并创建一个掩码来指示缺失的数据点。 add_artificial_mask 函数是 random_mask 函数的更复杂版本。它确保缺失值在整个序列中均匀分布。
- 该脚本使用 saving_into_h5 函数将预处理后的数据保存到HDF5文件中。 HDF5 文件是一种二进制文件格式，可有效存储大型数据集。

生成UCI电力消费数据集的脚本,主要逻辑是:

1. 读取csv格式的数据集文件。
2. 设置datetime索引,根据月份划分训练集、验证集和测试集。
3. 对特征做标准化处理。
4. 窗口截取生成样本。
5. 在训练集上人为引入一定比例的缺失。
6. 在训练集、验证集、测试集上加入人工缺失标记。
7. 保存为h5格式。
8. 保存标准化器。
9. 记录日志。
10. 打印数据集特征维度、样本数量等元信息。

与其他时间序列预测数据集生成脚本相比,主要区别在于:

1. 针对电力消费场景,使用月份划分数据。
2. 只在训练集引入人工缺失。
3. 打印并保存了各数据集样本数量占比。
4. 保存了标准化器,便于测试时反标准化。
通过参数配置,可以方便地生成不同时间长度、缺失率、划分方案的电力消费时间序列数据集。
