# 模型代码

这部分为比较核心，比较深入的内容，在理解代码的基础上，再去理解网络结构，便于我们针对网络结构来进行修改。

## 模型的基本结构，文件内容

C:\USERS\LUMISONG\DESKTOP\插补库\SAITS\MODELING
    brits.py
    layers.py
    mrnn.py
    saits.py
    transformer.py
    unified_dataloader.py
    utils.py

包含几个时间序列插补模型的实现：

brits.py - BRITS模型,基于双向RNN的时序插补模型
layers.py - 实现了一些基础层,如注意力层、门控循环单元(GRU)等
mrnn.py - 多层RNN模型,是一个简单的RNN baseline
saits.py - SAITS模型的实现
transformer.py - 基于Transformer的时序插补模型
unified_dataloader.py - 统一的数据加载器
utils.py - 一些公共的实用函数,如设置logger等

从文件可以看出,该代码库实现了不同类别的时序插补模型:

- 基于RNN的:BRITS, mrnn - 基于 RNN 的：BRITS、mrnn
- 基于注意力机制的:Transformer
- SAITS作为新的时序插补模型
通过统一的数据加载器和工具函数,不同模型可以重用代码、保持一致的训练、评估流程。

这是一个模块化、规范的时序插补建模代码库,通过不同模型的集成可以进行比较实验,也便于后续的模型开发。

