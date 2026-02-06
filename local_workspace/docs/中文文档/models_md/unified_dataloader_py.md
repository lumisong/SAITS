# unified_dataloader - 统一的数据加载器

## 原代码

```python

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def parse_delta(masks, seq_len, feature_num):
    """generate deltas from masks, used in BRITS"""
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas)


def fill_with_last_observation(arr):
    """namely forward-fill nan values
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    out = np.nan_to_num(out)  # if nan still exists then fill with 0
    return out


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf[set_name]["X"][:]
            self.X_hat = hf[set_name]["X_hat"][:]
            self.missing_mask = hf[set_name]["missing_mask"][:]
            self.indicating_mask = hf[set_name]["indicating_mask"][:]

        # fill missing values with 0
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ["Transformer", "SAITS"]:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype("float32")),
                torch.from_numpy(self.missing_mask[idx].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        elif self.model_type in ["BRITS", "MRNN"]:
            forward = {
                "X_hat": self.X_hat[idx],
                "missing_mask": self.missing_mask[idx],
                "deltas": parse_delta(
                    self.missing_mask[idx], self.seq_len, self.feature_num
                ),
            }
            backward = {
                "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
            }
            backward["deltas"] = parse_delta(
                backward["missing_mask"], self.seq_len, self.feature_num
            )
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward["X_hat"].astype("float32")),
                torch.from_numpy(forward["missing_mask"].astype("float32")),
                torch.from_numpy(forward["deltas"].astype("float32")),
                # for backward
                torch.from_numpy(backward["X_hat"].astype("float32")),
                torch.from_numpy(backward["missing_mask"].astype("float32")),
                torch.from_numpy(backward["deltas"].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        else:
            assert ValueError, f"Error model type: {self.model_type}"
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""

    def __init__(
        self, file_path, seq_len, feature_num, model_type, masked_imputation_task
    ):
        super(LoadTrainDataset, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
            assert (
                0 < self.artificial_missing_rate < 1
            ), "artificial_missing_rate should be greater than 0 and less than 1"

        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf["train"]["X"][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(
                indices,
                round(len(indices) * self.artificial_missing_rate),
            )
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            # reshape into time series
            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)

            if self.model_type in ["Transformer", "SAITS"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype("float32")),
                    torch.from_numpy(missing_mask.astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            elif self.model_type in ["BRITS", "MRNN"]:
                forward = {
                    "X_hat": X_hat,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }

                backward = {
                    "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward["X_hat"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    # for backward
                    torch.from_numpy(backward["X_hat"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            else:
                assert ValueError, f"Error model type: {self.model_type}"
        else:
            # if training without masked imputation task, then there is no need to artificially mask out observed values
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ["Transformer", "SAITS"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(missing_mask.astype("float32")),
                )
            elif self.model_type in ["BRITS", "MRNN"]:
                forward = {
                    "X": X,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }
                backward = {
                    "X": np.flip(forward["X"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward["X"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    # for backward
                    torch.from_numpy(backward["X"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                )
            else:
                assert ValueError, f"Error model type: {self.model_type}"
        return sample


class LoadDataForImputation(LoadDataset):
    """Load all data for imputation, we don't need do any artificial mask here,
    just input original data into models and let them impute missing values"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadDataForImputation, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf[set_name]["X"][:]
        self.missing_mask = (~np.isnan(self.X)).astype(np.float32)
        self.X = np.nan_to_num(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ["Transformer", "SAITS"]:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.missing_mask[idx].astype("float32")),
            )
        elif self.model_type in ["BRITS", "MRNN"]:
            forward = {
                "X": self.X[idx],
                "missing_mask": self.missing_mask[idx],
                "deltas": parse_delta(
                    self.missing_mask[idx], self.seq_len, self.feature_num
                ),
            }

            backward = {
                "X": np.flip(forward["X"], axis=0).copy(),
                "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
            }
            backward["deltas"] = parse_delta(
                backward["missing_mask"], self.seq_len, self.feature_num
            )
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward["X"].astype("float32")),
                torch.from_numpy(forward["missing_mask"].astype("float32")),
                torch.from_numpy(forward["deltas"].astype("float32")),
                # for backward
                torch.from_numpy(backward["X"].astype("float32")),
                torch.from_numpy(backward["missing_mask"].astype("float32")),
                torch.from_numpy(backward["deltas"].astype("float32")),
            )
        else:
            assert ValueError, f"Error model type: {self.model_type}"
        return sample


class UnifiedDataLoader:
    def __init__(
        self,
        dataset_path,
        seq_len,
        feature_num,
        model_type,
        batch_size=1024,
        num_workers=4,
        masked_imputation_task=False,
    ):
        """
        dataset_path: path of directory storing h5 dataset;
        seq_len: sequence length, i.e. time steps;
        feature_num: num of features, i.e. feature dimensionality;
        batch_size: size of mini batch;
        num_workers: num of subprocesses for data loading;
        model_type: model type, determine returned values;
        masked_imputation_task: whether to return data for masked imputation task, only for training/validation sets;
        """
        self.dataset_path = os.path.join(dataset_path, "datasets.h5")
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(
            self.dataset_path,
            self.seq_len,
            self.feature_num,
            self.model_type,
            self.masked_imputation_task,
        )
        self.val_dataset = LoadValTestDataset(
            self.dataset_path, "val", self.seq_len, self.feature_num, self.model_type
        )
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(
            self.dataset_path, "test", self.seq_len, self.feature_num, self.model_type
        )
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadDataForImputation(
            self.dataset_path, set_name, self.seq_len, self.feature_num, self.model_type
        )
        dataloader_for_imputation = DataLoader(
            data_for_imputation, self.batch_size, shuffle=False
        )
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation("train")
        val_set_for_imputation = self.prepare_dataloader_for_imputation("val")
        test_set_for_imputation = self.prepare_dataloader_for_imputation("test")
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation

```

## 代码解析

unified_dataloader.py 脚本旨在加载和预处理各种模型的训练、验证和测试数据，特别是用于在时间序列数据中插入缺失值的任务。该脚本旨在处理以 h5（分层数据格式版本 5）格式存储的数据集。

定义了几个类：

- LoadDataset ：用于加载数据集的超类。
- LoadValTestDataset ：用于加载验证和测试数据集的子类。
- LoadTrainDataset ：用于加载训练数据集的子类。它有一种处理屏蔽插补任务的附加方法，其中观察值的一部分被人为屏蔽以评估插补精度。
- LoadDataForImputation ：用于加载所有数据进行插补的子类。它不包括任何人为屏蔽数据；它只是将原始数据输入到模型中以估算缺失值。
- UnifiedDataLoader ：这是脚本中的主类，它结合了前面的所有类。

它加载用于训练、验证和测试的数据，并包括许多参数，例如序列长度（时间步数）、特征数量（特征空间的维数）和批量大小（特征空间的数量）。在更新模型的内部参数之前要完成的训练样本）。

在加载数据的过程中，应用了一些预处理步骤，包括用零填充缺失值（或者在训练数据的情况下用最后的观察值），生成“delta”值（指示自加载以来已经过了多少时间步）。最后一次观察），并将 numpy 数组转换为 PyTorch 张量。

根据用于插补的模型类型（例如 Transformer、SAITS、BRITS 或 MRNN）应用不同的处理规则。这些模型类型影响数据的排列和返回方式。

简而言之，该脚本是一个用于时间序列插补任务的综合数据预处理和加载实用程序，适合多种模型类型。

段代码实现了一个PyTorchDataset,用于加载和预处理时间序列数据用于训练不同类型的时序预测模型。
主要功能有:

1. LoadDataset类:定义了通用的数据加载逻辑,包含文件路径、序列长度、特征数量等信息。
2. LoadValTestDataset类:继承LoadDataset,实现验证和测试集的数据加载。主要是从h5文件读取数据,获取观测数据X、缺失数据X_hat、缺失标记missing_mask等。
3. LoadTrainDataset类:继承LoadDataset,实现训练集的数据加载。对观测数据人为遮蔽一部分作为X_hat,生成缺失标记missing_mask等。
4. parse_delta函数:用于BRITS和MRNN模型,从缺失标记生成delta向量。
5. fill_with_last_observation函数:用最后观测值填充缺失值。
6. __getitem__函数:根据模型类型返回不同格式的样本数据,包括forward和backward等。
7. 支持的模型包括:Transformer、SAITS、BRITS、MRNN等。
8. 可选择是否进行遮蔽补全(masked imputation)任务。

加载和预处理时间序列数据的通用DataLoader。主要功能有:

1. 支持加载训练、验证和测试集的数据。
2. 对于训练集,可以选择是否进行遮蔽补全(masked imputation)任务。如果进行该任务,会人为遮蔽部分数据作为需要预测的缺失值。
3. 支持多种模型格式的数据预处理,包括Transformer、SAITS、BRITS、MRNN等。不同模型需要的数据格式不同。
4. 提供了针对缺失值填充任务的数据加载接口,会直接加载原始数据,用于最后的模型预测填充。
5. 封装了PyTorch的DataLoader,支持批量读取数据。
6. 代码结构清晰,通过不同的Dataset类处理各个数据集,然后统一用DataLoader批量读取。
7. 支持配置不同参数,例如序列长度、特征维度、batch大小等。
8. 支持多进程数据读取以加速训练。
