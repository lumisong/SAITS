# transform 超参数优化过程

## 结构解析

Transformer_basic_config.ini 是运行Transformer模型的配置文件。 Transformer 模型是一类使用自注意力机制的模型，在许多自然语言处理任务中取得了成功。

Transformer_basic_config.ini 的关键部分和参数包括：

1. [file_path] ：指定保存模型、数据和结果的目录结构。
2. [dataset] ：指定正在使用的数据集（在本例中为电力消耗数据集）。它包括序列长度、特征数量、批量大小、工作人员数量和评估频率等参数。
3. [model] ：指定模型配置。在本例中，模型是一个具有多个参数的 Transformer，例如层组数 ( n_groups )、模型的隐藏维度 ( d_model ) 以及定义的其他参数模型的架构。
4. [training] ：指定训练参数，包括最大训练时期数、用于训练的设备、学习率和早期停止设置。
5. [test] ：指定测试设置，包括是否保存估算数据。

Transformer_searching_config.yml 是一个配置文件，用于运行 Transformer 模型的超参数搜索。它指定实验名称、作者姓名、可以同时运行的试验数量以及超参数的搜索空间。

Transformer_searching_space.json 提供了要搜索的超参数的空间。参数与 SAITS_searching_space.json 类似，例如组数 ( n_groups )、隐藏维度 ( d_model 、 d_inner ) 、注意力头的数量 ( n_head )、价值向量的大小 ( d_v ) 和丢失率 ( dropout )。学习率范围保持不变。

用于Transformer模型超参数搜索的:
Transformer_basic_config.ini: 定义了Transformer的基本训练配置,如数据集、模型保存路径等固定参数。
Transformer_searching_space.json: 定义了超参数的搜索空间,包括层数、大小、dropout比例、学习率等。
Transformer_searching_config.yml: 配置了搜索的设置,如优化算法、并发数、GPU资源等。
整体流程是:

- Transformer_basic_config.ini: 预设模型的固定基础训练参数
- Transformer_searching_space.json: 定义超参数的搜索空间
- Transformer_searching_config.yml: 配置搜索的参数
- 使用NNI的搜索优化算法探索最佳超参数
- 用最佳超参数配置更新Transformer_basic_config.ini作为固定配置进行模型训练
这样可以自动高效地搜索出Transformer在时序补缺任务上的最佳超参数,避免了人工反复试错,是NNI在时序模型超参优化上一个很好的实践
