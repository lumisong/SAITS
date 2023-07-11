# MRNN 调参

## 内容解析

MRNN_basic_config.ini 文件定义了用于训练、评估和测试 MRNN 模型（一种循环神经网络）的各种参数。该文件的结构几乎与 BRITS_basic_config.ini 相同，只有一些区别：

- [model] 下的 model_name 现在是 Electricity_MRNN ，表明此配置文件适用于在 electricity 数据集上训练的 MRNN 模型。
- model_type 现在是 MRNN ，反映了模型架构的变化。
- MRNN_searching_config.yml 文件与其对应的 BRITS_searching_config.yml 文件一样，使用 NNI 工具包设置超参数搜索实验，这次是针对 MRNN 模型。该文件的结构与 BRITS_searching_config.yml 完全相同，区别在于 trial 下的 command 指的是MRNN配置文件（ MRNN_basic_config.ini ）， searchSpacePath 指的是 MRNN 搜索空间文件（ MRNN_searching_space.json ）。
- 最后， MRNN_searching_space.json 指定要调整的超参数及其潜在值。参数的选择及其范围与 BRITS_searching_space.json 中的相同，这表明 MRNN 模型的调整过程将尝试使用相同范围的 RNN 层隐藏大小和学习率。

MRNN模型的基本配置文件、搜索空间文件和搜索算法配置文件,结构和BRITS的类似,也是用于利用NNI进行MRNN模型的自动超参数优化。
MRNN_basic_config.ini定义了MRNN的固定基本训练参数。
MRNN_searching_space.json定义了超参数搜索空间,这里搜索隐层大小rnn_hidden_size和学习率learning_rate。
MRNN_searching_config.yml配置了搜索的一些设置,如优化算法、并行trial数等。
整体流程是:

- MRNN_basic_config.ini: 预设模型固定基础参数
- MRNN_searching_space.json: 定义搜索空间
- MRNN_searching_config.yml: 配置搜索参数
- NNI搜索得到最优参数,更新到MRNN_basic_config.ini作为固定配置进行模型训练