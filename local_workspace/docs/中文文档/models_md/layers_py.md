# 网络层

Layer.py

PyTorch实现的Transformer encoder层相关模块。

## 实现网络的相关层

主要包含以下几个组件:

1. ScaledDotProductAttention:缩放点积注意力机制。
2. MultiHeadAttention:多头注意力机制,包含点积注意力以及线性映射。
3. PositionWiseFeedForward:位置全连接前馈网络。
4. EncoderLayer:Transformer编码器层,包含多头自注意力和前馈全连接。
5. PositionalEncoding:位置编码,通过正弦函数实现。

这个实现遵循Transformer的经典结构设计。
使用LayerNorm实现预处理归一化。
支持传入对角线遮挡的注意力掩码。
可配置是否使用因果注意力掩码。
提供了完整的多头自注意力计算流程。
可以 constit_query(k, v) 的形式用于其它任务。

- ScaledDotProductAttention：这是一个缩放的点积注意力层。它用于计算输入序列中不同位置之间的注意力权重。
- MultiHeadAttention：这是一个多头注意力层。它用于使用多个注意力头来计算输入序列中不同位置之间的注意力权重。
- PositionWiseFeedForward：这是一个位置前馈层。它用于向注意力层的输出添加非线性。
- EncoderLayer：这是一个编码器层。它由自注意力层、位置前馈层和残差连接组成。
- PositionalEncoding：这是一个位置编码层。它用于向输入序列添加位置信息。

这些是 Transformer 模型的层，Transformer 模型是一种常用于自然语言处理任务的深度学习模型，但也可以应用于时间序列分析等其他领域。

该代码定义了 Transformer 架构的几个关键组件，包括：

1. ScaledDotProductAttention ：这是注意力机制的一个组成部分，它帮助模型针对输出的每个部分关注输入的相关部分。此版本在应用 softmax 函数之前缩放查询和键的点积，这有助于深度模型中的梯度。
2. MultiHeadAttention ：这是一个并行多次应用注意力机制的模块，并对输入进行不同的学习线性变换。不同的“头”有可能学会关注输入中的不同特征。
3. PositionWiseFeedForward ：这是一个前馈神经网络，单独且相同地应用于每个位置。这包括两个线性变换，中间有一个 ReLU 激活。
4. EncoderLayer ：这是 Transformer 编码器的单层。它应用自注意力、丢失和位置前馈网络，并使用层归一化和残差连接。
5. PositionalEncoding ：该模块向输入添加位置编码，这有助于模型考虑序列中标记的顺序。位置编码是一种固定的正弦模式，它可以区分不同的位置，并允许模型泛化到以前从未见过的序列长度。

在原始 Transformer 模型中，多层 EncoderLayer 相互堆叠（multiple layers of EncoderLayer are stacked）以形成 Transformer Encoder。 Transformer Decoder 也具有类似的结构，但具有额外的交叉注意模块。additional cross-attention module.

这些模块共同构成了基于 Transformer 的架构的主要构建块，为 BERT、GPT 等更复杂的模型提供了基础。


