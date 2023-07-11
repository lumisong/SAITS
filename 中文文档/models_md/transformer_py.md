# transformer 基础模型


## 模型的基本结构，文件内容

```python

```

## 代码解析

适用于 TransformerEncoder 类，它也是一个 PyTorch 模块。它似乎是之前提供的 SAITS 模型的简化版本，使用标准转换器编码器结构对时间序列数据进行插补。它仍然实现了自我关注机制，但没有像 SAITS 模型中那样对数据进行第二次传递。

类的基本概要：

1. TransformerEncoder 类扩展了 PyTorch 的 nn.Module 。其中的自注意力机制类似于自然语言处理中使用的 Transformer 模型。
2. impute 方法用预测数据点替换时间序列中缺失的数据点。该模型利用时间序列中的可用数据及其自我关注机制来做出这些预测。
3. forward 方法计算原始数据点和估算数据点之间的平均绝对误差 (MAE)，以评估模型的性能。

与 SAITS 模型相比，该模型没有两个单独的预测阶段。它应用 Transformer 架构一次来预测并填充时间序列数据中的缺失值。

和之前一样， n_groups 、 n_group_inner_layers 、 d_model 、 n_head 等参数与标准 Transformer 模型中的参数非常相似，并且具有类似的目的。例如， n_head 指的是自注意力机制中注意力头的数量， d_model 表示输入嵌入的维度。

基于Transformer的时间序列填补模型的PyTorch实现。
主要包含以下几个部分:

1. TransformerEncoder 堆叠模块:包含多层多头注意力层。支持参数共享策略。
2. 特征嵌入和维度还原模块:处理时间和测量特征。
3. 前向过程:构建输入,调用填补方法,计算重构和填补损失。
实现了Transformer编码器结构。支持遮挡输入。
有不同的参数共享策略选择。
可以获得重构损失、填补损失、MAE等指标。

该模型由一堆编码器层组成，每个编码器层由自注意力层和前馈层组成。该模型已被证明在各种时间序列插补任务中都是有效的。
适用于 TransformerEncoder 模型。 TransformerEncoder 模型是基于 Transformer 架构的时间序列插补模型。