# SAITS 超参数优化

## 代码解析

SAITS_basic_config.ini 文件是自注意力插补和时间序列 (SAITS) 预测模型的配置文件。它提供有关用于训练、测试和配置模型的不同参数的信息。以下是一些值得注意的点：

1. 在 [model] 部分中，介绍了自注意力模型特有的几个参数，包括 n_groups （模型中层组的数量）、 n_group_inner_layers （每组内的层数）、 param_sharing_strategy （如何跨层共享参数）以及与模型架构相关的其他内容。
2. 模型类型定义为 SAITS ，代表自注意力插补和时间序列。
3. diagonal_attention_mask 参数指定是否应用对角注意掩模。该掩码可能会阻止模型关注序列中的未来位置，从而使其更适合时间序列数据。
4. [training] 部分有一个参数 masked_imputation_task ，该参数设置为 True ，表示模型将在训练期间执行插补任务。
5. 在 [test] 部分中， save_imputations 参数设置为 True ，这意味着模型将在测试阶段保存估算数据。

SAITS_searching_config.yml 文件是 SAITS 模型超参数搜索实验的配置文件。与之前的类似文件一样，它设置实验，定义运行试验的命令，并设置试验的各种参数。

最后， SAITS_searching_space.json 提供了要搜索的超参数的空间。超参数比以前更加复杂，包含与自注意力模型架构相关的参数，例如组数（ n_groups ）、隐藏维度（ d_model 、 d_inner ）、注意力头的数量（ n_head ）、价值向量的大小（ d_v ）和丢失率（ dropout ） 。学习率范围与以前的文件中的相同。

配置文件是用于SAITS模型的超参数搜索的:
SAITS_basic_config.ini:定义了SAITS模型的基本训练配置,如数据集、模型保存路径等固定的参数。
SAITS_searching_space.json:定义了要搜索的超参数空间,包括层数、隐层大小、dropout比例、学习率等。
SAITS_searching_config.yml:配置搜索的一些设置,如优化算法是随机搜索,并发数是12,使用的GPU等。
整体流程是:

- SAITS_basic_config.ini: 预设模型的固定基础训练参数
- SAITS_searching_space.json: 定义搜索的超参数空间
- SAITS_searching_config.yml: 配置搜索的一些设置
- 使用NNI的自动搜索,从定义的空间中探索出最佳超参数
- 用最佳超参数配置更新SAITS_basic_config.ini,进行模型的固定参数训练

