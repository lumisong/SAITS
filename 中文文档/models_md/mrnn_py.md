# mrnn

## 模型代码

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from modeling.brits import FeatureRegression
from modeling.utils import masked_mae_cal, masked_rmse_cal


class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super(FCN_Regression, self).__init__()
        self.feat_reg = FeatureRegression(rnn_hid_size * 2)
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x_t, m_t, target):
        h_t = F.tanh(
            F.linear(x_t, self.U * self.m)
            + F.linear(target, self.V1 * self.m)
            + F.linear(m_t, self.V2)
            + self.beta
        )
        x_hat_t = self.final_linear(h_t)
        return x_hat_t


class MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(MRNN, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = kwargs["device"]

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.rnn_cells = {"forward": self.f_rnn, "backward": self.b_rnn}
        self.concated_hidden_project = nn.Linear(
            self.rnn_hidden_size * 2, self.feature_num
        )
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, data, direction):
        values = data[direction]["X"]
        masks = data[direction]["missing_mask"]
        deltas = data[direction]["deltas"]

        hidden_states_collector = []
        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            hidden_state = self.rnn_cells[direction](inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def impute(self, data):
        hidden_states_f = self.gene_hidden_states(data, "forward")
        hidden_states_b = self.gene_hidden_states(data, "backward")[::-1]

        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]

        reconstruction_loss = 0
        estimations = []
        for i in range(
            self.seq_len
        ):  # calculating estimation loss for times can obtain better results than once
            x = values[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(
                x, m, RNN_imputed_data
            )  # FCN estimation is output extimation
            reconstruction_loss += masked_rmse_cal(
                FCN_estimation, x, m
            ) + masked_rmse_cal(RNN_estimation, x, m)
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [estimations, reconstruction_loss]

    def forward(self, data, stage):
        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]
        imputed_data, [estimations, reconstruction_loss] = self.impute(data)
        reconstruction_loss /= self.seq_len
        reconstruction_MAE = masked_mae_cal(estimations.detach(), values, masks)

        if stage == "val":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                imputed_data, data["X_holdout"], data["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        ret_dict = {
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "imputation_MAE": imputation_MAE,
            "imputed_data": imputed_data,
        }
        if "X_holdout" in data:
            ret_dict["X_holdout"] = data["X_holdout"]
            ret_dict["indicating_mask"] = data["indicating_mask"]
        return ret_dict

```

## 模型解析

文件包含两个类， FCN_Regression 和 MRNN 

- FCN_Regression 是作为 PyTorch nn.Module 实现的自定义回归模型。它使用似乎在另一个文件中实现的特征回归模块（feature regression module :self.feat_reg ）（此处未提供）。模型的参数包括 self.U 、 self.V1 、 self.V2 和 self.beta ，用于计算hidden states隐藏状态 h_t 的线性变换以产生输出 x_hat_t 。
- MRNN 是主要模型类。此类实现了一个掩码循环神经网络masked recurrent neural network ，用于处理可能丢失某些值的数据序列(sequences of data where some values may be missing)。它使用前向和后向 GRU ( nn.GRUCell ) 来处理两个方向的序列。
- gene_hidden_states 函数生成前向或后向序列的隐藏状态。 impute 函数使用前向和后向隐藏状态以及 fcn_regression （由 FCN_Regression 定义）模型来计算数据中缺失值的估计。
- 在 forward 函数中，模型计算估算数据和相应的重建损失。该函数还计算重建和插补阶段的平均绝对误差 (MAE) 指标。结果被打包成字典并返回。
- 最后，请注意 modeling.brits 和 modeling.utils 的导入表明该脚本是一个更大项目的一部分，并且类 FeatureRegression 、 masked_mae_cal 在代码库的其他地方定义。

MRNN 模型是一种双向递归神经网络 (RNN) 模型，用于估算时间序列数据中的缺失值。该模型由两个 RNN 组成，一个用于前向，一个用于后向。

RNN 经过训练可以预测输入序列中的缺失值。该模型还包括一个特征回归模块(feature regression module )，用于改进插补结果。

MRNN 模型能够学习时间序列数据中的远程依赖性。该模型还能够处理时间序列数据中的多个相关缺失值。该模型已被证明在各种时间序列插补任务中都是有效的。

基于MRNN的时间序列填补模型的PyTorch实现。

主要包含以下几个部分:

1. FCN_Regression:基于全连接网络的特征回归模块。
2. MRNN:主模型,包含前向和后向GRU,以及特征回归模块。
3. gene_hidden_states:生成前向和后向GRU的隐状态。
4. impute:进行时间步迭代,利用GRU隐状态和特征回归进行填补。
5. forward:组合各模块,完成训练/验证/测试流程。
实现了自定义的特征回归层。
支持前向和后向并行调用,增加模型容量。
可以获得重构损失、填补损失、MAE等指标。
