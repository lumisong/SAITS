# brits 双向RNN插补时间序列模型

Bi-directional Recurrent Imputation for Time Series (BRITS)

## 源代码

```python


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from modeling.utils import masked_mae_cal


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(RITS, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # other hyper parameters
        self.device = kwargs["device"]
        self.MIT = kwargs["MIT"]

        # create models
        self.rnn_cell = nn.LSTMCell(self.feature_num * 2, self.rnn_hidden_size)
        # # Temporal Decay here is used to decay the hidden state
        self.temp_decay_h = TemporalDecay(
            input_size=self.feature_num, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.feature_num, output_size=self.feature_num, diag=True
        )
        # # History regression and feature regression layer
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.feature_num)
        self.feat_reg = FeatureRegression(self.feature_num)
        # # weight-combine is used to combine history regression and feature regression
        self.weight_combine = nn.Linear(self.feature_num * 2, self.feature_num)

    def impute(self, data, direction):
        values = data[direction]["X"]
        masks = data[direction]["missing_mask"]
        deltas = data[direction]["deltas"]

        # use device of input values
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        estimations = []
        reconstruction_loss = 0.0
        reconstruction_MAE = 0.0

        # imputation period
        for t in range(self.seq_len):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += masked_mae_cal(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += masked_mae_cal(z_h, x, m)

            alpha = F.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_MAE += masked_mae_cal(c_h, x, m)
            reconstruction_loss += reconstruction_MAE

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [reconstruction_MAE, reconstruction_loss]

    def forward(self, data, direction="forward"):
        imputed_data, [reconstruction_MAE, reconstruction_loss] = self.impute(
            data, direction
        )
        reconstruction_MAE /= self.seq_len
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.seq_len * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=self.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputed_data": imputed_data,
        }
        if "X_holdout" in data:
            ret_dict["X_holdout"] = data["X_holdout"]
            ret_dict["indicating_mask"] = data["indicating_mask"]
        return ret_dict


class BRITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(BRITS, self).__init__()
        self.MIT = kwargs["MIT"]
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)
        self.rits_b = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def merge_ret(self, ret_f, ret_b, stage):
        consistency_loss = self.get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )
        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2
        reconstruction_loss = (
            ret_f["reconstruction_loss"] + ret_b["reconstruction_loss"]
        ) / 2
        reconstruction_MAE = (
            ret_f["reconstruction_MAE"] + ret_b["reconstruction_MAE"]
        ) / 2
        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                imputed_data, ret_f["X_holdout"], ret_f["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)
        imputation_loss = imputation_MAE

        ret_f["imputed_data"] = imputed_data
        ret_f["consistency_loss"] = consistency_loss
        ret_f["reconstruction_loss"] = reconstruction_loss
        ret_f["reconstruction_MAE"] = reconstruction_MAE
        ret_f["imputation_MAE"] = imputation_MAE
        ret_f["imputation_loss"] = imputation_loss
        return ret_f

    def impute(self, data):
        imputed_data_f, _ = self.rits_f.impute(data, "forward")
        imputed_data_b, _ = self.rits_b.impute(data, "backward")
        imputed_data_b = {"imputed_data_b": imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)["imputed_data_b"]
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data, [imputed_data_f, imputed_data_b]

    def forward(self, data, stage):
        ret_f = self.rits_f(data, "forward")
        ret_b = self.reverse(self.rits_b(data, "backward"))
        ret = self.merge_ret(ret_f, ret_b, stage)
        return ret
```

## 代码解析

时间序列双向循环插补 (BRITS) 的实现，这是一种使用循环神经网络 (RNN) 处理时间序列数据中缺失值的方法。

BRITS 方法由两个主要部分组成：

1. RITS：此类实现时间序列循环插补方法，该方法是 BRITS 方法的单向版本。它使用 LSTM 单元来更新 RNN 的隐藏状态。LSTM 单元将当前观测值（或上一步的估算观测值）与指示缺失值的掩码的组合作为输入。RITS 模型使用时间衰减机制，允许隐藏状态在远离已知观察时“忘记”信息。插补是基于观察特征回归的估计值和 RNN 输出的加权组合。
The LSTM cell takes as input the combination of the current observation (or the imputed observation from the previous step) and the mask that indicates missing values.

The RITS model uses a temporal decay mechanism that allows the hidden state to "forget" information as it moves further away from a known observation.

The imputation is a weighted combination of an estimated value based on the regression on observed features and the output of the RNN.
2. BRITS：此类将单向 RITS 扩展为双向设置。换句话说，它有两种RITS模型：一种是前向处理序列，另一种是后向处理序列。最终的插补是前向和后向 RITS 模型插补的平均值。

此实现还计算几个损失值：

1. 重建损失：这是观测条目的估算值与真实值之间的平均绝对误差。Reconstruction loss: This is the mean absolute error between the imputed value and the true value for the observed entries.
2. 一致性损失：这是一个正则化术语，用于确保前向和后向插补彼此一致。Consistency loss: This is a regularization term to ensure that the forward and backward imputations are consistent with each other.
3. 插补损失：这是插补值与保留集中的真实值之间的平均绝对误差。保留集是观察到的条目的一部分，在训练期间被视为缺失，用于评估插补性能。Imputation loss: This is the mean absolute error between the imputed values and the true values in the holdout set. The holdout set is a portion of the observed entries that is treated as missing during training, and is used to evaluate the imputation performance.

可以训练模型以最小化这些损失的总和。

BRITS 的双向性质使其能够在估算缺失值时考虑过去和未来时间步长的信息，与仅考虑过去信息的方法相比，可能会带来更好的性能。 consider information from both past and future time steps when imputing missing values.

BRITS 模型，它是时间序列数据的双向循环插补模型。该模型由两种 RITS 模型组成，一种用于前向插补，一种用于后向插补。然后合并两个 RITS 模型以生成最终的估算数据。

BRITS模式具有以下优点：

- 它可以有效地处理时间序列数据中的远程依赖关系。
- 它可以同时在向前和向后方向上估算缺失值。
- 它可用于为多个时间序列数据集训练单个模型。

BRITS 模型已被证明在各种时间序列插补任务中都是有效的。例如，它已被用来估算财务数据、医疗数据和传感器数据中的缺失值。

提供的代码是 BRITS 模型的 PyTorch 实现。代码组织良好且易于理解。它还包括许多有用的注释，解释了代码的不同部分。

这是BRITS模型的pytorch实现,主要包含以下部分:

1. FeatureRegression: 用于特征回归的全连接层。
2. TemporalDecay: 对隐状态进行时间衰减的层。
3. RITS: 单向RNN时序插补模块,包含历史回归、特征回归等。
4. BRITS: 双向BRITS模型,包含正向和反向RITS,并计算前后一致性损失。

模型的forward流程是:

1. 正向和反向各通过一个RITS模块进行推理,分别得到正向和反向的插补结果。
2. 计算正向反向结果的一致性损失。
3. 结合正向反向结果得到最终插补数据。
4. 输出插补数据、重构损失、一致性损失等。

实现了自定义的TemporalDecay层、特征回归层。
利用统一的数据接口,可以灵活配置是否使用缺失率掩码进行半监督训练。
