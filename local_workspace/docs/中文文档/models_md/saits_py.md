# saits 模型

Self-Attention Interpolation Time Series (SAITS)

## 模型的基本结构，文件内容

```python
from modeling.layers import *
from modeling.utils import masked_mae_cal

class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = (
            masks * X + (1 - masks) * X_tilde_3
        )  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                X_tilde_3, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
```

## 代码解析

代码似乎是自注意力插值时间序列（SAITS）模型的实现。它专为时间序列数据插补而设计。这是一个基本轮廓：

self-attention mechanism

1. SAITS 类扩展了 PyTorch 的 nn.Module 并实现了自注意力机制，这与自然语言处理中使用的 Transformer 模型非常相似。在这种情况下，注意力机制用于更好地对时间序列数据中的依赖关系进行建模。
2. impute 方法用预测数据点替换时间序列中缺失的数据点。该模型使用时间序列中的可用数据及其自注意力机制来做出这些预测。The impute method replaces missing data points in the time series with the predicted ones. The model uses the available data in a time series and its self-attention mechanism to make these predictions.
3. forward 方法计算原始数据点和估算数据点之间的平均绝对误差 (MAE)，以衡量模型的性能。calculates the Mean Absolute Error (MAE) between the original and imputed data points

值得注意的是，自注意力机制被使用了两次，每次的目标都略有不同：

a.在第一个块中，它用于仅使用可用数据来估计缺失值 ( X_tilde_1 )。In the first block, it's used to estimate missing values (X_tilde_1) using only available data.
b.在第二个块中，考虑到第一个块中已填充的间隙，它用于细化初始估计 ( X_tilde_2 )。. In the second block, it's used to refine the initial estimation (X_tilde_2) considering the already filled-in gaps from the first block.

然后将这两个估计值结合起来 ( X_tilde_3 ) 以创建对缺失数据的更准确的预测。

n_groups 、 n_group_inner_layers 、 d_model 、 n_head 等模型配置参数与典型 Transformer 模型中的参数非常相似，并提供服务类似的目的。例如， n_head 指的是自注意力机制中注意力头的数量， d_model 指的是输入嵌入的维度。The model configuration parameters like n_groups, n_group_inner_layers, d_model, n_head, etc., closely resemble those in a typical Transformer model and serve similar purposes. For example, n_head refers to the number of attention heads in the self-attention mechanism, and d_model refers to the dimensionality of the input embeddings.

SAITS模型是基于DMSA算法的时间序列插补模型。由两个 DMSA 块组成，第二个 DMSA 块的输出使用注意力加权组合函数与第一个 DMSA 块的输出组合。该模型已被证明在各种时间序列插补任务中都是有效的。

基于自注意力的时间序列填补(SAITS)模型的PyTorch实现。
主要包含以下几个部分:

1. MSA 堆叠模块:包含多层多头自注意力层。支持参数共享策略。
2. 特征嵌入和维度还原模块:处理时间和测量特征。
3. 注意力加权组合模块:结合两个MSA块的输出。
4. 前向过程:构建输入,调用填补方法,计算重构和填补损失。
实现了双MSA块结构。支持遮挡输入。
有不同的参数共享策略选择。
可以获得重构损失、填补损失、MAE等指标。
