# 针对于Brits 模型的调参，使用nni进行调参

## 文件

三个文件：BRITS_basic_config.ini、BRITS_searching_config.yml、BRITS_searching_space.json

## 文件内容

BRITS_basic_config.ini 文件定义用于训练、评估和测试时间序列双向循环插补 (BRITS) 模型的各种参数。一些值得注意的参数是：

- 数据集、模型文件和图形的保存路径。
- 特定于数据集的参数，例如序列长度、特征数量、批量大小等。
- BRITS 模型特有的参数，例如 RNN 的隐藏大小。
- 训练参数，例如是否包含屏蔽插补任务、epoch 数、学习率等。
- 测试参数，例如是否保存插补以及特定模型文件和保存图形的目录的路径。

BRITS_searching_config.yml 文件是神经网络智能 (NNI) 工具包用于超参数调整的配置文件。它指定实验名称、作者、并发级别、训练服务平台、搜索空间文件的路径以及调谐器的类型（在本例中为随机搜索）。它还包括每次试验要运行的命令、代码目录以及使用的 GPU 数量。

BRITS_searching_space.json 文件定义了要调整的超参数以及它们可以采用的可能值。在这里，它正在调整两个超参数：

1. rnn_hidden_size ：循环神经网络层中隐藏单元的数量。调谐器（tuner）将随机选择（randomly select）指定大小之一：32、64、128、256、512 或 1024。
2. learning_rate ：优化器的学习率。将从 0.0001 到 0.01 之间的对数均匀分布中随机选择一个值。

此过程将允许 NNI 对给定参数执行超参数搜索，这将有助于找到 BRITS 模型的最佳配置。

BRITS模型的基本配置文件、搜索空间文件和搜索算法配置文件,用于利用NNI进行BRITS模型的自动超参数优化。
BRITS_basic_config.ini定义了BRITS的基本训练配置,如数据集路径、模型保存路径、Early Stopping等。这是固定的预设配置。
BRITS_searching_space.json定义了搜索时的参数空间,这里搜索隐层大小rnn_hidden_size和学习率learning_rate。
BRITS_searching_config.yml配置了搜索的一些元参数,如优化算法是随机搜索,同时搜索的Trial数等。
整体来说,这组配置文件的作用是:

- BRITS_basic_config.ini: 定义模型固定的基础训练参数
- BRITS_searching_space.json: 定义搜索的参数空间
- BRITS_searching_config.yml: 配置搜索的参数
- 通过NNI调优后找到最优的参数,更新到BRITS_basic_config.ini中进行模型固定的训练