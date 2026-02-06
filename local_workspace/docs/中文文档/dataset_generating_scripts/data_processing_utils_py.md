# 数据处理通用脚本

操作和处理数据（处理缺失值的情况下）

## 总体介绍

| 函数名称 | 功能介绍 | 简述 |
| :--- | :--- | :--- |
| window_truncate() | 时间序列数据截断为给定长度的序列 | 窗口截取 |
| random_mask() | 生成随机掩码的索引 | |
| add_artificial_mask() | 向时间序列数据添加人工缺失值 | |
| saving_into_h5() | 将处理后的数据保存到HDF5文件中 | 保存到h5文件 |

基本说明：
    1. window_truncate：该函数通过从给定序列长度的时间序列数据中截断窗口来生成时间序列样本。它使用滑动窗口来提取时间序列的子序列。如果未提供 sliding_len ，则将其设置为 seq_len ，这意味着窗口之间没有重叠。
    2. random_mask：此函数生成随机掩码的索引。它需要一个向量和一个人工缺失率，然后随机选择一个索引子集，其中的值将替换为 NaN（非数字）以模拟缺失数据。
    3. add_artificial_mask：此函数将人工缺失值添加到输入特征向量中。它创建输入 (X_hat) 的副本，并随机选择索引子集设置为 NaN。对于训练数据，它还计算经验平均值，该平均值用于某些时间序列插补方法。处理后的数据在带有键“X”、“X_hat”、“missing_mask”和“indicating_mask”的字典中返回。
    4. saving_into_h5：该函数将数据保存到.h5文件（一种分层数据格式文件格式）中，以便以后可以高效地加载。它在文件中为训练、验证和测试数据创建不同的数据集。

时间序列数据处理的通用工具函数:

1. window_truncate(): 从时间序列中截取固定长度的窗口作为样本。
2. random_mask(): 随机遮蔽时间序列中的一些值,以生成人为丢失。
3. add_artificial_mask(): 为时间序列添加人为丢失,分训练集、验证集、测试集处理逻辑不同。
4. saving_into_h5(): 将处理后的时间序列样本保存到h5文件。

主要功能包括:

- 窗口截取
- 加入人为丢失
- 计算缺失值填充的经验均值
- 生成mask
- 保存到h5文件

## 函数细节

### window_truncate

```python
def window_truncate(feature_vectors, seq_len, sliding_len=None):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    sliding_len: size of the sliding window
    """
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if total_len - start_indices[-1] * sliding_len < seq_len:  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])
    return np.asarray(sample_collector).astype('float32')
```

window_truncate 函数用于从给定的时间序列数据集创建时间序列样本。它通过创建数据滑动窗口来实现这一点，每个窗口的长度为 seq_len 。

以下是该函数功能的详细说明：

1. 如果未提供 sliding_len ，则将其设置为 seq_len ，这意味着窗口之间没有重叠。
2. total_len 设置为时间序列数据的长度 ( feature_vectors.shape[0] )。
3. 函数计算 start_indices ，这是每个窗口的起点。这是通过创建从 0 到 total_len // sliding_len 的数字范围（适合总长度的完整窗口的数量），然后将该范围乘以 sliding_len 来完成的。这这确保了每个起始索引与下一个索引的距离为 slip_len。
4. 如果最后一个窗口之后的剩余部分短于 seq_len ，则删除最后一个起始索引以确保所有窗口具有相同的长度。
5. 对于每个起始索引，该函数从时间序列数据中提取长度为 seq_len 的窗口并将其附加到 sample_collector 。
6. 该函数返回转换为“float32”类型的 NumPy 数组的 sample_collector 。

综上所述， window_truncate 从时间序列数据集中提取特定长度的滑动窗口，可以作为时间序列建模的样本。

window_truncate() 函数有两个参数：

- feature_vectors ：时间序列数据，是形状 [total_length, feature_num] 的 NumPy 数组
- seq_len ：要生成的序列的长度

首先通过将时间序列数据的总长度除以序列长度来计算可以生成的序列总数。然后，它生成序列的起始索引列表。起始索引均匀分布，间距为 sliding_len 。然后，该函数从指定起始索引处的时间序列数据中提取序列。序列作为形状 **[num_samples, seq_len, feature_num]** 的**NumPy 数组**返回。

时间序列数据的长度为100，序列长度为20，则该函数将生成5个序列。序列的起始索引将为 0、20、40、60 和 80。将从这些起始索引处的时间序列数据中提取序列。

window_truncate函数，主要逻辑:

A. 滑动窗口的步长sliding_len,如果不指定则默认等于序列长度seq_len。
B. 计算可以截取的起始索引start_indices,通过总长度除以滑动窗口大小,间隔sliding_len取值。
C. 去掉最后一个索引,如果其超出总长度会导致截取不到完整窗口。
D. 遍历起始索引,截取从每个索引开始长度为seq_len的序列窗口,保存到样本集合sample_collector中。
E. 样本集转换为numpy数组并返回。

这种固定窗口截取是处理时间序列数据的一种常用方法,可以将长序列分割成固定维度的样本进行训练和处理。
这个函数提供了一种通用的实现,通过输入时间序列和参数就可以批量生成所需的窗口样本。

### random_mask

```python
def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices
```

`random_mask` 函数生成掩码索引，该索引将用于人为地将缺失值引入数据中。

工作原理如下：

1. 使用两个参数：`vector` 和 `artificial_missing_rate`。`vector` 是一个一维的 numpy 数组，`artificial_missing_rate` 是一个介于 0 和 1 之间的浮点数，表示要随机屏蔽的值的比例。
2. 使用 `assert len(vector.shape) == 1` 来断言 `vector` 只有一个维度。如果 `vector` 有多个维度，函数将引发 AssertionError。
3. 语句 `np.where(~np.isnan(vector))[0].tolist()` 识别 `vector` 中值不是 NaN（即不缺失）的索引。`~` 符号是一个逻辑 NOT 操作，表示“不是”。因此，`~np.isnan(vector)` 给出一个布尔数组，其中 `vector` 的值不是 NaN 的位置为 True。`[0]` 获取 `np.where()` 返回的元组的第一个元素，`tolist()` 将结果索引数组转换为列表。
4. `np.random.choice(indices, int(len(indices) * artificial_missing_rate))` 随机选择这些索引的子集。所选索引的数量由 `artificial_missing_rate` 决定，因此如果 `artificial_missing_rate` 为 0.1，则将选择 10% 的索引。
5. 函数返回所选的索引。

该函数从向量中随机选择一些非缺失值的子集，这些值将在后面被替换为 NaN，以有效地创建人工缺失数据。缺失数据的比例由 `artificial_missing_rate` 控制。

random_mask() 函数用于生成随机掩码的索引。该函数有两个参数：

- vector ：一维的 NumPy 数组
- artificial_missing_rate ：将要创建的人为缺失值的比率。

通过将时间序列数据的长度乘以人工缺失率来计算需要生成的缺失值总数。然后，它生成缺失值的索引列表。

索引是从时间序列数据中的非缺失值中随机选择的。该函数返回索引列表。

例如，如果时间序列数据的长度为 100，人工缺失率为 0.1，则该函数将生成 10 个缺失值。缺失值的索引将从时间序列数据中的 90 个非缺失值中随机选择。

生成人工丢失的随机索引。

输入一个向量vector和人工丢失比率artificial_missing_rate。

主要逻辑：

1. 首先断言向量vector必须是一维的。
2. 使用np.where找到向量中不是NaN的索引indices。
3. 从indices中随机选取int(len(indices) * artificial_missing_rate)个索引。
4. 返回选取的索引。

这样就可以随机在向量中的实值上生成一定比例的丢失值,来模拟真实场景中的缺失情况。

返回的随机索引可以用于在向量对应的位置替换为NaN来产生丢失。

### artificial_missing

```python

def add_artificial_mask(X, artificial_missing_rate, set_name):
    """Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X,
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),
        }

    return data_dict
```

`add_artificial_mask` 用于将人工缺失值添加到数据集中，这对于测试处理缺失数据的算法非常有用。该函数区分训练集和验证/测试集，对每个集应用不同的过程。

1. 该函数采用三个参数： `X` 、 `artificial_missing_rate` 和 `set_name` 。 `X` 是一个包含特征向量的 3D numpy 数组， `artificial_missing_rate` 是一个浮点数，表示要随机屏蔽的数据比例， `set_name` 是一个字符串，表示是否数据是训练集、验证集或测试集。
2. 输入数据 `X` 的形状被解包为 `sample_num` 、 `seq_len` 和 `feature_num` 。
3. 如果 `set_name` 为“train”，则该函数计算每个特征在所有时间点和值不缺失的样本上的经验平均值，稍后将由 GRU-D 模型使用。结果平均值存储在 `empirical_mean_for_GRUD` 数组中。
    1. 该函数首先生成一个掩码，指示 `X` 中哪些值不是 `NaN`。
    2. 然后，它将 `X` 中的任何 `NaN` 值替换为零，从而得到 `X_filledWith0` 。
    3. 经验平均值的计算方法是将 `X_filledWith0` 之和乘以掩码（其效果是仅对非 `NaN` 值求和），然后除以掩码之和（给出非 `NaN` 值的计数） -`NaN` 值）。
    4. 最后，创建一个包含 `X` 和 `empirical_mean_for_GRUD` 的 `data_dict` 字典。
4. 如果 `set_name` 不是“train”（即，它是验证集或测试集），则该函数会生成人工缺失值：
    1. 该函数首先将 `X` 重塑为一维数组。
    2. 然后，它调用 `random_mask` 函数来获取应丢失的值的索引。这些索引存储在 `indices_for_holdout` 中。
    3. 生成 `X` 的副本，命名为 `X_hat` ，并用 `NaN` 替换 `indices_for_holdout` 指定的索引处的值。
    4. 创建一个 `missing_mask` 来指示 `X_hat` 不为 `NaN`。
    5. 创建一个 `indicating_mask` 来显示添加了人工缺失值的位置。它通过在 `X_hat` 不为 `NaN` 和 `X` 不为 `NaN` 之间执行逻辑 `XOR` 运算来实现此目的。
    6. 最后，创建一个 `data_dict` 字典，其中包含 `X` 、 `X_hat` 、 `missing_mask` 和 `indicating_mask` ，所有内容都 `reshaped` 回来变成 `[sample_num, seq_len, feature_num]` 的原始形状。
5. 该函数返回 `data_dict` 。

简而言之，该函数向数据集引入了人工缺失值，并且可以灵活地处理训练和验证/测试数据集。对于想要了解不同方法如何处理缺失数据的实验来说，这一点非常重要。

add_artificial_mask() 函数用于向时间序列数据添加人工缺失值。该函数采用三个参数：

- X ：时间序列数据，是形状 [total_length, feature_num] 的 NumPy 数组。
- artificial_missing_rate ：将要创建的人为缺失值的比率。
- set_name ：数据集的名称，可以是 train 、 val 或 test 。

首先检查 set_name 参数的值。如果值为 train ，则该函数不会向时间序列数据添加任何人为缺失值。这是因为 SAITS 模型可以在训练过程中学习估算缺失值，而向训练数据中添加人工缺失值会使训练过程变得更加困难。

set_name 参数的值为 val 或 test ，则该函数会向时间序列数据添加人工缺失值。该函数首先生成缺失值的索引列表。然后，它将指定索引处的值设置为 NaN 。该函数还创建两个掩码： missing_mask 和 indicating_mask 。 missing_mask 指示值是否缺失。 indicating_mask 表示原始时间序列数据中是否存在缺失值，或者人工缺失值的时间序列数据中是否存在缺失值。

dd_artificial_mask函数用于为时间序列数据添加人工丢失。
主要逻辑:

1. 计算时间序列X的形状。
2. 对训练集,不需要提前添加丢失,训练时通过丢弃随机遮蔽方式实现。计算并保存GRU-D模型需要的经验均值。
3. 对验证集和测试集,需要提前添加丢失。调用random_mask随机生成丢失索引,将X对应的位置设为NaN得到X_hat。
4. 生成遮蔽标记missing_mask和丢失指示标记indicating_mask。
5. 保存原始数据X,带丢失数据X_hat和生成的遮蔽标记到字典中返回。

这样通过在验证集和测试集提前添加人工丢失,可以用于评估模型的插补效果。
对训练集的处理方式可以灵活引入随机丢弃遮蔽,增强模型的泛化能力。

### saving_into_h5

```python

def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset("labels", data=data["labels"].astype(int))
        single_set.create_dataset("X", data=data["X"].astype(np.float32))
        if name in ["val", "test"]:
            single_set.create_dataset("X_hat", data=data["X_hat"].astype(np.float32))
            single_set.create_dataset(
                "missing_mask", data=data["missing_mask"].astype(np.float32)
            )
            single_set.create_dataset(
                "indicating_mask", data=data["indicating_mask"].astype(np.float32)
            )

    saving_path = os.path.join(saving_dir, "datasets.h5")
    with h5py.File(saving_path, "w") as hf:
        hf.create_dataset(
            "empirical_mean_for_GRUD",
            data=data_dict["train"]["empirical_mean_for_GRUD"],
        )
        save_each_set(hf, "train", data_dict["train"])
        save_each_set(hf, "val", data_dict["val"])
        save_each_set(hf, "test", data_dict["test"])

```

函数 saving_into_h5 用于将数据保存到HDF5（分层数据格式版本5）文件中。 HDF5 是一种用于存储和管理数据的数据模型、库和文件格式，广泛用于存储大量数值数据（如数组）。

函数的详细说明：

1. 该函数采用三个参数： saving_dir 、 data_dict 和 classification_dataset 。 saving_dir 是一个字符串，指定数据保存的路径。 data_dict 是一个字典，包含要保存的数据， classification_dataset 是一个布尔标志，指示数据集是否用于分类。
2. 嵌套函数 save_each_set 是在 saving_into_h5 内定义的。此函数采用三个参数： handle 、 name 和 data 。 handle 是要保存数据的HDF5文件或组， name 是数据集的名称， data 是要保存的数据。
   1. 该函数首先在 HDF5 文件或 handle 指定的组中创建一个新组，其名称由 name 指定。
   2. 如果 classification_dataset 为 True，它会在组中创建一个名为“labels”的新数据集，数据由 data["labels"] 指定，并将数据转换为整数类型。
   3. 然后，它创建一个名为“X”的新数据集和 data["X"] 指定的数据，并将数据转换为 float32 类型。
   4. 如果 name 为“val”或“test”，它还会创建名称为“X_hat”、“missing_mask”和“indicating_mask”的数据集，并将相应数据转换为float32类型。
3. 保存数据的完整路径是通过连接 saving_dir 和字符串“datasets.h5”创建的。
4. 该函数在指定路径打开或创建具有写权限的 HDF5 文件。
5. 在文件内部，它创建一个名为“empirical_mean_for_GRUD”的数据集和 `data_dict["train"]["empirical_mean_for_GRUD"]` 指定的数据。
6. 然后，它调用 save_each_set 函数将训练集、验证集和测试集保存到 HDF5 文件中。
7. 所有数据写入后，退出 with 块时文件自动关闭。

总之，此功能有助于有效存储大量数值数据以供进一步处理或实验。

它根据数据的性质（训练、验证或测试）将数据整齐地组织到不同的组中，并在保存之前将它们转换为适当的数据类型。

函数有两个参数：

- saving_dir ：保存 HDF5 文件的目录路径。
- data_dict ：包含已处理数据的数据字典。

该函数首先在指定目录中创建一个HDF5文件。然后，它迭代三组数据：train、val 和 test。对于每组，该函数将数据保存到 HDF5 文件中。

数据保存在单独的数据集中，每个数据集对应于数据的不同特征。例如， X 数据集包含时间序列数据， missing_mask 数据集包含缺失值掩码， indicating_mask 数据集包含指示掩码。

saving_into_h5函数用于将处理后的时间序列数据集保存到h5文件中。
主要逻辑:

1. 定义save_each_set内部函数,根据名称将单个数据集分别保存到h5文件中,包含X、标签、丢失的X_hat、遮蔽等。
2. 拼接保存路径。
3. 打开h5文件,保存GRU-D需要的经验均值。
4. 分别调用内部函数保存训练集、验证集、测试集。
这样可以将数据集的划分结果完整地保存到h5文件,方便后续读取使用。
使用h5格式可以高效存储数组数据,COMPONENTS容易组织多维数组,并可以压缩存储。
提供了将处理后时间序列数据集存储到磁盘的简单高效实现。
