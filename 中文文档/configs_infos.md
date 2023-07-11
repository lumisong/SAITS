# 文件结构

AirQuality_BRITS_best.ini
AirQuality_MRNN_best.ini
AirQuality_SAITS_base.ini
AirQuality_SAITS_best.ini
AirQuality_Transformer_best.ini
Electricity_BRITS_best.ini
Electricity_MRNN_best.ini
Electricity_SAITS_base.ini
Electricity_SAITS_best.ini
Electricity_Transformer_best.ini
NRTSI_AirQuality_SAITS_best.ini
NRTSI_Gas_SAITS_best.ini
PhysioNet2012_BRITS_best.ini
PhysioNet2012_MRNN_best.ini
PhysioNet2012_SAITS_base.ini
PhysioNet2012_SAITS_best.ini
PhysioNet2012_Transformer_best.ini

## 文件说明

1. 这些都是不同数据集和模型的最佳配置文件,存储在configs目录下。
2. 数据集包括：
   1. AirQuality：UCI 北京空气质量数据集
   2. Electricity：UCI 电力负荷数据集
   3. NRTSI：NRTSI 2018数据集，包括空气质量和天然气数据集；实时流数据,包含AirQuality和Gas两个子数据集
   4. PhysioNet2012：Physionet2012心电图数据集
3. 模型包括：
   1. BRITS：Bidirectional Recurrent Imputation for Time Series, 2018;双向循环插补模型
   2. MRNN：Multi-directional Recurrent Neural Network，2019；多方向循环神经网络
   3. SAITS：Self-Attention based Imputation and Time Series, 2023；基于自注意力的插补和时间序列模型
   4. Transformer：Attention is All You Need, 2017；注意力机制模型
4. 细节：
   1. airquality：四个模型的最佳配置文件+一个SAITS的基础配置文件
   2. electricity：四个模型的最佳配置文件+一个SAITS的基础配置文件
   3. NRTSI：SAITS的最佳配置文件（两个子数据集），空气质量和天然气
   4. PhysioNet2012：四个模型的最佳配置文件+一个SAITS的基础配置文件

## 文件细节

### AirQuality_BRITS_best.ini

空气质量数据集上BRITS模型的最佳配置文件

#### 配置文件内容

```ini
[file_path]
; prefix of saving dir
prefix = .
; base dir, in which the dataset is saved
dataset_base_dir = generated_datasets
result_saving_base_dir = NIPS_results

; Below items are for testing
; dir to save models
model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-05-04_T18:16:55
; dir to save graphs, which will be plotted in model testing
test_results_saving_base_dir = ${prefix}/NIPS_results

[dataset]
dataset_name = AirQuality_seqlen24_01masked
seq_len = 24
feature_num = 132
batch_size = 128
num_workers = 4
eval_every_n_steps = 7

[model]
; name of your model, will be the name of dir to save your models and logs
model_name = AirQuality_BRITS_best
; model type
model_type = BRITS
; hidden size of RNN
rnn_hidden_size = 1024

[training]
; whether to have Masked Imputation Task (MIT) in training
MIT = False
; whether to have Observed Reconstruction Task (ORT) in training
ORT = True
; max num of training epochs
epochs = 10000
; which device for training, cpu/cuda
device = cuda
; learning rate
lr = 0.00015385579939014617
; weight for reconstruction loss
reconstruction_loss_weight = 1
; weight for imputation loss
imputation_loss_weight = 1
; weight for consistency loss, here we use to adjust the importance of consistency loss
consistency_loss_weight = 1
; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
early_stop_patience = 30
; what type of optimizer to use, adam/adamw
optimizer_type = adam
; weight decay used in optimizer
weight_decay = 0
; max_norm for gradient clipping, set 0 to disable
max_norm = 0
; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
model_saving_strategy = best

[test]
; whether to save imputed data
save_imputations = True
; name of model your select for testing
step_675 = model_trainStep_4725_valStep_675_imputationMAE_0.1403
; absolute path to locate model you select
model_path = ${file_path:model_saving_dir}/${step_675}
; path of dir to save generated figs (PR-curve etc.)
result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_675

```

分为以下几个部分：<a id="ini文件结构"></a>

1. `[file_path]` : 指定各种类型文件的目录，包括数据集、结果、模型和测试结果。变量用于创建这些路径， `${variable}`语法引用文件不同部分的变量。
2. `[dataset]` ：配置要使用的数据集的详细信息，包括数据集名称、序列长度、特征数量、批量大小、`number of workers`以及何时评估。
3. `[model]` ：包含特定于模型的参数，例如模型名称、类型和 RNN 的隐藏大小。
4. `[training]` ：设置训练模型的参数。包括是否在训练中使用某些类型任务的标志 `(Masked Imputation Task (MIT) and Observed Reconstruction Task (ORT))`（掩码插补任务 (MIT) 和观察重建任务 (ORT)）、训练时期的最大数量`maximum number of training epochs`、用于训练的设备（CPU 或用于 GPU 的 CUDA）、学习率、不同类型损失的权重（重建、插补、一致性）`reconstruction, imputation, consistency`、提前停止的设置、要使用的优化器类型、优化器中的权重衰减量`amount of weight decay in the optimizer`、梯度裁剪最大范数`gradient clipping max norm`和模型保存策略`model saving strategy`。
5. `[test]` ：配置测试模型的设置。它决定是否保存插补、选择用于测试的模型的名称、定位所选模型的路径以及保存生成的图形（如 PR 曲线）的路径。

详细说明：

- 目录部分

    ```ini
    ; prefix of saving dir
    prefix = .
    ; base dir, in which the dataset is saved
    dataset_base_dir = generated_datasets
    result_saving_base_dir = NIPS_results

    ; Below items are for testing
    ; dir to save models
    model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-05-04_T18:16:55
    ; dir to save graphs, which will be plotted in model testing
    test_results_saving_base_dir = ${prefix}/NIPS_results

    ```

  - `prefix`：保存目录的前缀，用于创建保存目录的绝对路径。
  - `dataset_base_dir`：数据集的基本目录，用于创建数据集的绝对路径。
  - `result_saving_base_dir`：结果保存的基本目录，用于创建结果保存的绝对路径。
  - `model_saving_dir`：保存模型的目录，用于创建保存模型的绝对路径。组成部分包括`prefix`、`result_saving_base_dir`、`model:model_name`和`models/2021-05-04_T18:16:55`。使用`model:model_name`是为了在保存模型时使用模型名称，而`models/2021-05-04_T18:16:55`是为了在保存模型时使用时间戳。
  - `test_results_saving_base_dir`：保存测试结果的基本目录，用于创建保存测试结果的绝对路径。

注意：

1. `prefix`、`dataset_base_dir`和`result_saving_base_dir`是必需的，因为它们用于创建数据集和结果保存的绝对路径。
2. `model_saving_dir`和`test_results_saving_base_dir`是可选的，因为它们仅用于测试模型。
3. windows 系统下，路径分隔符为`\`，而 linux 系统下，路径分隔符为`/`。
4. windows 系统下，最后运行的文件保存在`/`目录下，而 linux 系统下，最后运行的文件保存在`/home`目录下。windows根目录为`C:\`

- 数据集部分

    ```ini
    [dataset]
    dataset_name = AirQuality_seqlen24_01masked
    seq_len = 24
    feature_num = 132
    batch_size = 128
    num_workers = 4
    eval_every_n_steps = 7
    ```

  - `dataset_name`：数据集名称
  - `seq_len`：序列长度
  - `feature_num`：特征数量
  - `batch_size`：批量大小，用于创建数据加载器
  - `num_workers`：`number of workers`，用于创建数据加载器
  - `eval_every_n_steps`：每隔多少步评估一次模型

- 模型部分

    ```ini
    [model]
    ; name of your model, will be the name of dir to save your models and logs
    model_name = AirQuality_BRITS_best
    ; model type
    model_type = BRITS
    ; hidden size of RNN
    rnn_hidden_size = 1024
    ```

  - `model_name`：模型名称，将用于创建保存模型和日志的目录
  - `model_type`：模型类型，包括`BRITS`、`LSTM`、`GRU`和`RNN`。
  - `rnn_hidden_size`：RNN 的隐藏大小

- 训练部分

    ```ini
    [training]
    ; whether to have Masked Imputation Task (MIT) in training
    MIT = False
    ; whether to have Observed Reconstruction Task (ORT) in training
    ORT = True
    ; max num of training epochs
    epochs = 10000
    ; which device for training, cpu/cuda
    device = cuda
    ; learning rate
    lr = 0.00015385579939014617
    ; weight for reconstruction loss
    reconstruction_loss_weight = 1
    ; weight for imputation loss
    imputation_loss_weight = 1
    ; weight for consistency loss, here we use to adjust the importance of consistency loss
    consistency_loss_weight = 1
    ; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
    early_stop_patience = 30
    ; what type of optimizer to use, adam/adamw
    optimizer_type = adam
    ; weight decay used in optimizer
    weight_decay = 0
    ; max_norm for gradient clipping, set 0 to disable
    max_norm = 0
    ; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
    model_saving_strategy = best
    ```

  - `MIT`：是否在训练中使用掩码插补任务
  - `ORT`：是否在训练中使用观察重构任务
  - `epochs`：最大训练轮数
  - `device`：训练设备，包括`cpu`和`cuda`
  - `lr`：学习率, 0.00015385579939014617
  - `reconstruction_loss_weight`：重构损失的权重, 1
  - `imputation_loss_weight`：插补损失的权重, 1
  - `consistency_loss_weight`：一致性损失的权重, 1; 用于调整一致性损失的重要性.
  - `early_stop_patience`：早停的耐心，-1表示不应用（当前早停是基于总损失的）, 30
  - `optimizer_type`：优化器类型，包括`adam`和`adamw`, adam
  - `weight_decay`：优化器中的权重衰减, 0
  - `max_norm`：梯度裁剪的最大范数，设置为0表示禁用, 0
  - `model_saving_strategy`：模型保存策略，包括`all`、`best`和`none`。如果设置为`none`，则不保存模型（超参数搜索模式）, best

- 测试部分

    ```ini
    [test]
    ; whether to save imputed data
    save_imputations = True
    ; name of model your select for testing
    step_675 = model_trainStep_4725_valStep_675_imputationMAE_0.1403
    ; absolute path to locate model you select
    model_path = ${file_path:model_saving_dir}/${step_675}
    ; path of dir to save generated figs (PR-curve etc.)
    result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_675
    ```

  - `save_imputations`：是否保存插补数据
  - `step_675`：选择用于测试的模型的名称
  - `model_path`：定位所选模型的绝对路径
  - `result_saving_path`：保存生成的图形（PR-curve等）的目录的路径

总结：部分名称说明

1. NIPS——神经信息处理系统（Neural Information Processing Systems）
2. BRITS——双向循环插补时间序列（BRITS: Bidirectional Recurrent Imputation for Time Series）
3. step_675——步数为675
4. model_trainStep_4725_valStep_675_imputationMAE_0.1403——模型名称，训练步数为4725，验证步数为675，插补 MAE 为 0.1403

### AirQuality_MRNN_best.ini

空气质量数据集的 MRNN 模型的配置文件。

#### 配置文件内容_AirQuality_MRNN_best.ini

```ini
[file_path]
; prefix of saving dir
prefix = .
; base dir, in which the dataset is saved
dataset_base_dir = generated_datasets
result_saving_base_dir = NIPS_results

; Below items are for testing
; dir to save models
model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-28_T03:44:08
; dir to save graphs, which will be plotted in model testing
test_results_saving_base_dir = ${prefix}/NIPS_results

[dataset]
dataset_name = AirQuality_seqlen24_01masked
seq_len = 24
feature_num = 132
batch_size = 128
num_workers = 4
eval_every_n_steps = 7

[model]
; name of your model, will be the name of dir to save your models and logs
model_name = AirQuality_MRNN_best
; model type
model_type = MRNN
; hidden size of RNN
rnn_hidden_size = 256

[training]
; whether to have Masked Imputation Task (MIT) in training
MIT = False
; whether to have Observed Reconstruction Task (ORT) in training
ORT = True
; max num of training epochs
epochs = 10000
; which device for training, cpu/cuda
device = cuda
; learning rate
lr = 0.0009238749874174623
; weight for reconstruction loss
reconstruction_loss_weight = 1
; weight for imputation loss
imputation_loss_weight = 1
; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
early_stop_patience = 30
; what type of optimizer to use, adam/adamw
optimizer_type = adam
; weight decay used in optimizer
weight_decay = 0
; max_norm for gradient clipping, set 0 to disable
max_norm = 0
; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
model_saving_strategy = best

[test]
; whether to save imputed data
save_imputations = True
; name of model your select for testing
step_183 = model_trainStep_1281_valStep_183_imputationMAE_0.1984
; absolute path to locate model you select
model_path = ${file_path:model_saving_dir}/${step_183}
; path of dir to save generated figs (PR-curve etc.)
result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_183
```

分为以下几个部分：

同上，不再赘述。

- 路径部分

  ```ini
  [file_path]
  ; prefix of saving dir
  prefix = .
  ; base dir, in which the dataset is saved
  dataset_base_dir = generated_datasets
  result_saving_base_dir = NIPS_results
  
  ; Below items are for testing
  ; dir to save models
  model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-28_T03:44:08
  ; dir to save graphs, which will be plotted in model testing
  test_results_saving_base_dir = ${prefix}/NIPS_results
  ```

  - `prefix`：保存目录的前缀
  - `dataset_base_dir`：保存数据集的基本目录
  - `result_saving_base_dir`：保存结果的基本目录
  - `model_saving_dir`：保存模型的目录
  - `test_results_saving_base_dir`：保存测试结果的目录

- 数据集部分

    ```ini
    [dataset]
    dataset_name = AirQuality_seqlen24_01masked
    seq_len = 24
    feature_num = 132
    batch_size = 128
    num_workers = 4
    eval_every_n_steps = 7
    ```

  - `dataset_name`：数据集名称
  - `seq_len`：序列长度
  - `feature_num`：特征数量
  - `batch_size`：批大小
  - `num_workers`：工作进程数
  - `eval_every_n_steps`：每隔多少步进行一次评估

- 模型部分

    ```ini
    [model]
    ; name of your model, will be the name of dir to save your models and logs
    model_name = AirQuality_MRNN_best
    ; model type
    model_type = MRNN
    ; hidden size of RNN
    rnn_hidden_size = 256
    ```

  - `model_name`：模型名称
  - `model_type`：模型类型
  - `rnn_hidden_size`：RNN 隐藏层大小

- 训练部分

    ```ini
    [training]
    ; whether to have Masked Imputation Task (MIT) in training
    MIT = False
    ; whether to have Observed Reconstruction Task (ORT) in training
    ORT = True
    ; max num of training epochs
    epochs = 10000
    ; which device for training, cpu/cuda
    device = cuda
    ; learning rate
    lr = 0.0009238749874174623
    ; weight for reconstruction loss
    reconstruction_loss_weight = 1
    ; weight for imputation loss
    imputation_loss_weight = 1
    ; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
    early_stop_patience = 30
    ; what type of optimizer to use, adam/adamw
    optimizer_type = adam
    ; weight decay used in optimizer
    weight_decay = 0
    ; max_norm for gradient clipping, set 0 to disable
    max_norm = 0
    ; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
    model_saving_strategy = best

    ```

  - `MIT`：是否在训练中使用 MIT
  - `ORT`：是否在训练中使用 ORT
  - `epochs`：最大训练轮数
  - `device`：训练设备
  - `lr`：学习率
  - `reconstruction_loss_weight`：重构损失权重
  - `imputation_loss_weight`：填充损失权重
  - `early_stop_patience`：早停等待轮数
  - `optimizer_type`：优化器类型
  - `weight_decay`：优化器权重衰减
  - `max_norm`：梯度裁剪阈值
  - `model_saving_strategy`：模型保存策略

- 测试部分

    ```ini
    [test]
    ; whether to save imputed data
    save_imputations = True
    ; name of model your select for testing
    step_183 = model_trainStep_1281_valStep_183_imputationMAE_0.1984
    ; absolute path to locate model you select
    model_path = ${file_path:model_saving_dir}/${step_183}
    ; path of dir to save generated figs (PR-curve etc.)
    result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_183
    ```

  - `save_imputations`：是否保存填充数据
  - `step_183`：测试步数
  - `model_path`：模型路径
  - `result_saving_path`：结果保存路径

总结：部分文件名称说明

1. generated_datasets：生成的数据集
2. AirQuality_seqlen24_01masked：数据集名称，sqllen 为 24，缺失率为 0.1
3. AirQuality_MRNN_best：模型名称，MRNN 模型
4. model_trainStep_1281_valStep_183_imputationMAE_0.1984：模型名称，训练步数为 1281，验证步数为 183，填充 MAE 为 0.1984

### AirQuality_SAITS_base.ini

空气质量数据集的 SAITS 模型的基本配置文件。

#### 配置文件内容——AirQuality_SAITS_base.ini

```ini
[file_path]
; prefix of saving dir
prefix = .
; base dir, in which the dataset is saved
dataset_base_dir = generated_datasets
result_saving_base_dir = NIPS_results

; Below items are for testing
; dir to save models
model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-21_T23:37:45
; dir to save graphs, which will be plotted in model testing
test_results_saving_base_dir = ${prefix}/NIPS_results

[dataset]
dataset_name = AirQuality_seqlen24_01masked
seq_len = 24
feature_num = 132
batch_size = 128
num_workers = 4
eval_every_n_steps = 7

[model]
; name of your model, will be the name of dir to save your models and logs
model_name = AirQuality_SAITS_base
; whether concat input with missing mask
input_with_mask = True
; model type, Transformer/SAITS
model_type = SAITS
; num of layer groups
n_groups = 2
; num of group-inner layers
n_group_inner_layers = 1
; how to share parameters, inner_group/between_group
param_sharing_strategy = inner_group
; model hidden dim
d_model = 256
; hidden size of feed forward layer
d_inner = 128
; head num of self-attention
n_head = 4
; key dim
d_k = 64
; value dim
d_v = 64
; drop out rate
dropout = 0.1
; whether to apply diagonal attention mask
diagonal_attention_mask = True

[training]
; whether to have Masked Imputation Task (MIT) in training
MIT = True
; whether to have Observed Reconstruction Task (ORT) in training
ORT = True
; max num of training epochs
epochs = 10000
; which device for training, cpu/cuda
device = cuda
; learning rate
lr = 0.001
; weight for reconstruction loss
reconstruction_loss_weight = 1
; weight for imputation loss
imputation_loss_weight = 1
; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
early_stop_patience = 30
; what type of optimizer to use, adam/adamw
optimizer_type = adam
; weight decay used in optimizer
weight_decay = 0
; max_norm for gradient clipping, set 0 to disable
max_norm = 0
; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
model_saving_strategy = best

[test]
; whether to save imputed data
save_imputations = True
; name of model your select for testing
step_505 = model_trainStep_3535_valStep_505_imputationMAE_0.1360
; absolute path to locate model you select
model_path = ${file_path:model_saving_dir}/${step_505}
; path of dir to save generated figs (PR-curve etc.)
result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_505
```

- 文件路径部分

    ```ini
    [file_path]
    ; prefix of saving dir
    prefix = .
    ; base dir, in which the dataset is saved
    dataset_base_dir = generated_datasets
    result_saving_base_dir = NIPS_results

    ; Below items are for testing
    ; dir to save models
    model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-21_T23:37:45
    ; dir to save graphs, which will be plotted in model testing
    test_results_saving_base_dir = ${prefix}/NIPS_results
    ```

  - `prefix`：保存目录前缀
  - `dataset_base_dir`：数据集保存目录
  - `result_saving_base_dir`：结果保存目录
  - `model_saving_dir`：模型保存目录
  - `test_results_saving_base_dir`：测试结果保存目录

- 数据集部分

    ```ini
    [dataset]
    dataset_name = AirQuality_seqlen24_01masked
    seq_len = 24
    feature_num = 132
    batch_size = 128
    num_workers = 4
    eval_every_n_steps = 7
    ```

  - `dataset_name`：数据集名称
  - `seq_len`：序列长度
  - `feature_num`：特征数量
  - `batch_size`：批大小
  - `num_workers`：数据加载器的进程数量
  - `eval_every_n_steps`：每隔多少步进行一次评估

- 模型部分

    ```ini
    [model]
    ; name of your model, will be the name of dir to save your models and logs
    model_name = AirQuality_SAITS_base
    ; whether concat input with missing mask
    input_with_mask = True
    ; model type, Transformer/SAITS
    model_type = SAITS
    ; num of layer groups
    n_groups = 2
    ; num of group-inner layers
    n_group_inner_layers = 1
    ; how to share parameters, inner_group/between_group
    param_sharing_strategy = inner_group
    ; model hidden dim
    d_model = 256
    ; hidden size of feed forward layer
    d_inner = 128
    ; head num of self-attention
    n_head = 4
    ; key dim
    d_k = 64
    ; value dim
    d_v = 64
    ; drop out rate
    dropout = 0.1
    ; whether to apply diagonal attention mask
    diagonal_attention_mask = True
    ```

  - `model_name`：模型名称
  - `input_with_mask`：输入是否包含缺失掩码
  - `model_type`：模型类型，Transformer/SAITS
  - `n_groups`：分组数量
  - `n_group_inner_layers`：分组内层数量
  - `param_sharing_strategy`：参数共享策略，inner_group/between_group
  - `d_model`：模型隐藏层维度
  - `d_inner`：前馈层隐藏层维度
  - `n_head`：自注意力头数
  - `d_k`：键维度
  - `d_v`：值维度
  - `dropout`：Dropout概率
  - `diagonal_attention_mask`：是否应用对角注意力掩码

- 训练部分

    ```ini
    [training]
    ; whether to have Masked Imputation Task (MIT) in training
    MIT = True
    ; whether to have Observed Reconstruction Task (ORT) in training
    ORT = True
    ; max num of training epochs
    epochs = 10000
    ; which device for training, cpu/cuda
    device = cuda
    ; learning rate
    lr = 0.001
    ; weight for reconstruction loss
    reconstruction_loss_weight = 1
    ; weight for imputation loss
    imputation_loss_weight = 1
    ; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
    early_stop_patience = 30
    ; what type of optimizer to use, adam/adamw
    optimizer_type = adam
    ; weight decay used in optimizer
    weight_decay = 0
    ; max_norm for gradient clipping, set 0 to disable
    max_norm = 0
    ; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
    model_saving_strategy = best
    ```

  - `MIT`：是否在训练中使用缺失掩码预测任务
  - `ORT`：是否在训练中使用重构任务
  - `epochs`：最大训练轮数
  - `device`：训练设备，cpu/cuda
  - `lr`：学习率
  - `reconstruction_loss_weight`：重构损失权重
  - `imputation_loss_weight`：缺失掩码预测损失权重
  - `early_stop_patience`：早停等待轮数，-1表示不使用早停（当前早停基于总损  ）
  - `optimizer_type`：优化器类型，adam/adamw
  - `weight_decay`：优化器权重衰减
  - `max_norm`：梯度裁剪最大范数，设置为0表示不使用梯度裁剪
  - `model_saving_strategy`：模型保存策略，all/best/none。如果设置为none，则不保存模型（用于超参数搜索）,best表示保存最好的模型，all表示保存所有模型

- 测试部分

    ```ini
    [test]
    ; whether to save imputed data
    save_imputations = True
    ; name of model your select for testing
    step_505 = model_trainStep_3535_valStep_505_imputationMAE_0.1360
    ; absolute path to locate model you select
    model_path = ${file_path:model_saving_dir}/${step_505}
    ; path of dir to save generated figs (PR-curve etc.)
    result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_505
    ```

  - `save_imputations`：是否保存缺失值填充结果
  - `step_505`：选择的模型名称
  - `model_path`：选择的模型路径
  - `result_saving_path`：结果保存路径

总结：

1. AirQuality_seqlen24_01masked:空气质量数据集，序列长度为24，缺失率为0.1
2. AirQuality_SAITS_base:SAITS模型，分组数量为2，分组内层数量为1，参数共享策略为inner_group，模型隐藏层维度为256，前馈层隐藏层维度为128，自注意力头数为4，键维度为64，值维度为64，Dropout概率为0.1，是否应用对角注意力掩码为True
3. step_505:选择的模型名称
4. model_trainStep_3535_valStep_505_imputationMAE_0.1360:模型名称，训练步数为3535，验证步数为505，缺失值填充MAE为0.1360

### AirQuality_SAITS_best.ini

空气质量数据集在SAITS模型上的最优配置文件

#### 配置文件内容——AirQuality_SAITS_best.ini

```ini
[file_path]
; prefix of saving dir
prefix = .
; base dir, in which the dataset is saved
dataset_base_dir = generated_datasets
result_saving_base_dir = NIPS_results

; Below items are for testing
; dir to save models
model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-28_T19:46:52
; dir to save graphs, which will be plotted in model testing
test_results_saving_base_dir = ${prefix}/NIPS_results

[dataset]
dataset_name = AirQuality_seqlen24_01masked
seq_len = 24
feature_num = 132
batch_size = 128
num_workers = 4
eval_every_n_steps = 7

[model]
; name of your model, will be the name of dir to save your models and logs
model_name = AirQuality_SAITS_best
; whether concat input with missing mask
input_with_mask = True
; model type, Transformer/SAITS
model_type = SAITS
; num of layer groups
n_groups = 1
; num of group-inner layers
n_group_inner_layers = 1
; how to share parameters, inner_group/between_group
param_sharing_strategy = inner_group
; model hidden dim
d_model = 512
; hidden size of feed forward layer
d_inner = 512
; head num of self-attention
n_head = 4
; key dim
d_k = 128
; value dim
d_v = 64
; drop out rate
dropout = 0
; whether to apply diagonal attention mask
diagonal_attention_mask = True

[training]
; whether to have Masked Imputation Task (MIT) in training
MIT = True
; whether to have Observed Reconstruction Task (ORT) in training
ORT = True
; max num of training epochs
epochs = 10000
; which device for training, cpu/cuda
device = cuda
; learning rate
lr = 0.00088213879506932665
; weight for reconstruction loss
reconstruction_loss_weight = 1
; weight for imputation loss
imputation_loss_weight = 1
; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
early_stop_patience = 30
; what type of optimizer to use, adam/adamw
optimizer_type = adam
; weight decay used in optimizer
weight_decay = 0
; max_norm for gradient clipping, set 0 to disable
max_norm = 0
; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
model_saving_strategy = best

[test]
; whether to save imputed data
save_imputations = True
; name of model your select for testing
step_527 = model_trainStep_3689_valStep_527_imputationMAE_0.1221
; absolute path to locate model you select
model_path = ${file_path:model_saving_dir}/${step_527}
; path of dir to save generated figs (PR-curve etc.)
result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_527
```

- 文件路径部分

  ```ini
  [file_path]
  ; prefix of saving dir
  prefix = .
  ; base dir, in which the dataset is saved
  dataset_base_dir = generated_datasets
  result_saving_base_dir = NIPS_results
  
  ; Below items are for testing
  ; dir to save models
  model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-28_T19:46:52
  ; dir to save graphs, which will be plotted in model testing
  test_results_saving_base_dir = ${prefix}/NIPS_results
  ```

  - `prefix`：保存目录前缀
  - `dataset_base_dir`：数据集保存目录
  - `result_saving_base_dir`：结果保存目录
  - `model_saving_dir`：模型保存目录
  - `test_results_saving_base_dir`：测试结果保存目录

- 数据集部分

    ```ini
    [dataset]
    dataset_name = AirQuality_seqlen24_01masked
    seq_len = 24
    feature_num = 132
    batch_size = 128
    num_workers = 4
    eval_every_n_steps = 7
    ```

  - `dataset_name`：数据集名称
  - `seq_len`：序列长度
  - `feature_num`：特征数量
  - `batch_size`：批大小
  - `num_workers`：数据加载器的进程数量
  - `eval_every_n_steps`：每隔多少步进行一次验证

- 模型部分

    ```ini
    [model]
    ; name of your model, will be the name of dir to save your models and logs
    model_name = AirQuality_SAITS_best
    ; whether concat input with missing mask
    input_with_mask = True
    ; model type, Transformer/SAITS
    model_type = SAITS
    ; num of layer groups
    n_groups = 1
    ; num of group-inner layers
    n_group_inner_layers = 1
    ; how to share parameters, inner_group/between_group
    param_sharing_strategy = inner_group
    ; model hidden dim
    d_model = 512
    ; hidden size of feed forward layer
    d_inner = 512
    ; head num of self-attention
    n_head = 4
    ; key dim
    d_k = 128
    ; value dim
    d_v = 64
    ; drop out rate
    dropout = 0
    ; whether to apply diagonal attention mask
    diagonal_attention_mask = True
    ```

  - `model_name`：模型名称
  - `input_with_mask`：输入是否包含缺失掩码
  - `model_type`：模型类型，Transformer/SAITS
  - `n_groups`：分组数
  - `n_group_inner_layers`：分组内层数
  - `param_sharing_strategy`：参数共享策略，inner_group/between_group
  - `d_model`：模型隐藏层维度
  - `d_inner`：前馈层隐藏层维度
  - `n_head`：自注意力头数
  - `d_k`：键维度
  - `d_v`：值维度
  - `dropout`：Dropout概率
  - `diagonal_attention_mask`：是否应用对角注意力掩码

- 训练部分

    ```ini
    [training]
    ; whether to have Masked Imputation Task (MIT) in training
    MIT = True
    ; whether to have Observed Reconstruction Task (ORT) in training
    ORT = True
    ; max num of training epochs
    epochs = 10000
    ; which device for training, cpu/cuda
    device = cuda
    ; learning rate
    lr = 0.00088213879506932665
    ; weight for reconstruction loss
    reconstruction_loss_weight = 1
    ; weight for imputation loss
    imputation_loss_weight = 1
    ; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
    early_stop_patience = 30
    ; what type of optimizer to use, adam/adamw
    optimizer_type = adam
    ; weight decay used in optimizer
    weight_decay = 0
    ; max_norm for gradient clipping, set 0 to disable
    max_norm = 0
    ; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
    model_saving_strategy = best
    ```

  - `MIT`：是否在训练中使用掩码填充任务
  - `ORT`：是否在训练中使用观测重构任务
  - `epochs`：最大训练轮数
  - `device`：训练设备，cpu/cuda
  - `lr`：学习率
  - `reconstruction_loss_weight`：重构损失权重
  - `imputation_loss_weight`：填充损失权重
  - `early_stop_patience`：早停等待轮数，-1表示不使用（当前早停是基于总损失的）
  - `optimizer_type`：优化器类型，adam/adamw
  - `weight_decay`：优化器权重衰减
  - `max_norm`：梯度裁剪的最大范数，设置为0表示不使用
  - `model_saving_strategy`：模型保存策略，all/best/none。如果设置为none，则不保存模型（用于超参数搜索）

- 测试部分

    ```ini
    [test]
    ; whether to save imputed data
    save_imputations = True
    ; name of model your select for testing
    step_527 = model_trainStep_3689_valStep_527_imputationMAE_0.1221
    ; absolute path to locate model you select
    model_path = ${file_path:model_saving_dir}/${step_527}
    ; path of dir to save generated figs (PR-curve etc.)
    result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_527
    ```

  - `save_imputations`：是否保存填充后的数据
  - `step_527`：选择用于测试的模型名称
  - `model_path`：选择用于测试的模型的绝对路径
  - `result_saving_path`：保存生成的图形（PR曲线等）的目录路径

总结：

1. AirQuality_seqlen24_01masked：序列长度为24，缺失率为0.1，缺失掩码为1
2. AirQuality_SAITS_best：SAITS模型，最佳模型
3. model_trainStep_3689_valStep_527_imputationMAE_0.1221：训练轮数3689，验证轮数527，填充MAE为0.1221

### AirQuality_Transformer_best.ini

基于基础transformer的配置文件，用于训练基础transformer模型。

#### 配置文件内容_基础transformer

```ini
[file_path]
; prefix of saving dir
prefix = .
; base dir, in which the dataset is saved
dataset_base_dir = generated_datasets
result_saving_base_dir = NIPS_results

; Below items are for testing
; dir to save models
model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-22_T02:45:33
; dir to save graphs, which will be plotted in model testing
test_results_saving_base_dir = ${prefix}/NIPS_results

[dataset]
dataset_name = AirQuality_seqlen24_01masked
seq_len = 24
feature_num = 132
batch_size = 128
num_workers = 4
eval_every_n_steps = 7

[model]
; name of your model, will be the name of dir to save your models and logs
model_name = AirQuality_Transformer_best
; whether concat input with missing mask
input_with_mask = True
; model type, Transformer/SAITS
model_type = Transformer
; num of layer groups
n_groups = 1
; num of group-inner layers
n_group_inner_layers = 1
; how to share parameters, inner_group/between_group
param_sharing_strategy = inner_group
; model hidden dim
d_model = 1024
; hidden size of feed forward layer
d_inner = 1024
; head num of self-attention
n_head = 8
; key dim
d_k = 128
; value dim
d_v = 32
; drop out rate
dropout = 0.1
; whether to apply diagonal attention mask
diagonal_attention_mask = False

[training]
; whether to have Masked Imputation Task (MIT) in training
MIT = True
; whether to have Observed Reconstruction Task (ORT) in training
ORT = True
; max num of training epochs
epochs = 10000
; which device for training, cpu/cuda
device = cuda
; learning rate
lr = 0.0004290163334443633
; weight for reconstruction loss
reconstruction_loss_weight = 1
; weight for imputation loss
imputation_loss_weight = 1
; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
early_stop_patience = 30
; what type of optimizer to use, adam/adamw
optimizer_type = adam
; weight decay used in optimizer
weight_decay = 0
; max_norm for gradient clipping, set 0 to disable
max_norm = 0
; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
model_saving_strategy = best

[test]
; whether to save imputed data
save_imputations = True
; name of model your select for testing
step_387 = model_trainStep_2709_valStep_387_imputationMAE_0.1433
; absolute path to locate model you select
model_path = ${file_path:model_saving_dir}/${step_387}
; path of dir to save generated figs (PR-curve etc.)
result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_387
```

- 文件路径

  ```ini
  [file_path]
  ; prefix of saving dir
  prefix = .
  ; base dir, in which the dataset is saved
  dataset_base_dir = generated_datasets
  result_saving_base_dir = NIPS_results
  ; Below items are for testing
  ; dir to save models
  model_saving_dir = ${prefix}/${result_saving_base_dir}/${model:model_name}/models/2021-04-22_T02:45:33
  ; dir to save graphs, which will be plotted in model testing
  test_results_saving_base_dir = ${prefix}/NIPS_results
  ```

  - `prefix`：保存目录的前缀
  - `dataset_base_dir`：数据集所在的基础目录
  - `result_saving_base_dir`：结果保存的基础目录
  - `model_saving_dir`：保存模型的目录
  - `test_results_saving_base_dir`：保存测试结果的目录

- 数据集

  ```ini
  [dataset]
  dataset_name = AirQuality_seqlen24_01masked
  seq_len = 24
  feature_num = 132
  batch_size = 128
  num_workers = 4
  eval_every_n_steps = 7
  ```

  - `dataset_name`：数据集名称
  - `seq_len`：序列长度
  - `feature_num`：特征数量
  - `batch_size`：批大小
  - `num_workers`：工作线程数
  - `eval_every_n_steps`：每隔多少步进行一次评估

- 模型

    ```ini
    [model]
    ; name of your model, will be the name of dir to save your models and logs
    model_name = AirQuality_Transformer_best
    ; whether concat input with missing mask
    input_with_mask = True
    ; model type, Transformer/SAITS
    model_type = Transformer
    ; num of layer groups
    n_groups = 1
    ; num of group-inner layers
    n_group_inner_layers = 1
    ; how to share parameters, inner_group/between_group
    param_sharing_strategy = inner_group
    ; model hidden dim
    d_model = 1024
    ; hidden size of feed forward layer
    d_inner = 1024
    ; head num of self-attention
    n_head = 8
    ; key dim
    d_k = 128
    ; value dim
    d_v = 32
    ; drop out rate
    dropout = 0.1
    ; whether to apply diagonal attention mask
    diagonal_attention_mask = False
    ```

  - `model_name`：模型名称
  - `input_with_mask`：输入是否包含缺失掩码
  - `model_type`：模型类型，Transformer/SAITS
  - `n_groups`：层组数
  - `n_group_inner_layers`：层组内层数
  - `param_sharing_strategy`：参数共享策略，inner_group/between_group
  - `d_model`：模型隐藏维度
  - `d_inner`：前馈层隐藏维度
  - `n_head`：自注意力头数
  - `d_k`：键维度
  - `d_v`：值维度
  - `dropout`：dropout率
  - `diagonal_attention_mask`：是否应用对角注意力掩码

- 训练

    ```ini
    [training]
    ; whether to have Masked Imputation Task (MIT) in training
    MIT = True
    ; whether to have Observed Reconstruction Task (ORT) in training
    ORT = True
    ; max num of training epochs
    epochs = 10000
    ; which device for training, cpu/cuda
    device = cuda
    ; learning rate
    lr = 0.0004290163334443633
    ; weight for reconstruction loss
    reconstruction_loss_weight = 1
    ; weight for imputation loss
    imputation_loss_weight = 1
    ; patience of early stopping, -1 means not applied (current early stopping is based on total loss)
    early_stop_patience = 30
    ; what type of optimizer to use, adam/adamw
    optimizer_type = adam
    ; weight decay used in optimizer
    weight_decay = 0
    ; max_norm for gradient clipping, set 0 to disable
    max_norm = 0
    ; strategy on model saving, all/best/none. If set as none, then do not save models (mode for hyper-parameter searching)
    model_saving_strategy = best
    ```

  - `MIT`：是否在训练中使用MIT
  - `ORT`：是否在训练中使用ORT
  - `epochs`：最大训练轮数
  - `device`：训练设备，cpu/cuda
  - `lr`：学习率
  - `reconstruction_loss_weight`：重构损失权重
  - `imputation_loss_weight`：填充损失权重
  - `early_stop_patience`：早停等待轮数，-1表示不使用（当前早停是基于总损失的）
  - `optimizer_type`：优化器类型，adam/adamw
  - `weight_decay`：优化器中的权重衰减
  - `max_norm`：梯度裁剪的最大范数，设置为0表示不使用
  - `model_saving_strategy`：模型保存策略，all/best/none。如果设置为none，则不保存模型（用于超参数搜索）
  
- 测试部分

    ```ini
    [test]
    ; whether to save imputed data
    save_imputations = True
    ; name of model your select for testing
    step_387 = model_trainStep_2709_valStep_387_imputationMAE_0.1433
    ; absolute path to locate model you select
    model_path = ${file_path:model_saving_dir}/${step_387}
    ; path of dir to save generated figs (PR-curve etc.)
    result_saving_path = ${file_path:test_results_saving_base_dir}/${model:model_name}/step_387
    ```

  - `save_imputations`：是否保存填充后的数据
  - `step_387`：选择用于测试的模型名称
  - `model_path`：选择用于测试的模型的绝对路径
  - `result_saving_path`：保存生成的图形（PR曲线等）的目录路径

总结：

1. AirQuality_seqlen24_01masked：缺失率为0.1，序列长度为24的数据集
2. model_trainStep_2709_valStep_387_imputationMAE_0.1433：训练2709轮，验证387轮，填充MAE为0.1433的模型。

## 整体信息

整体文件的配置信息参数，使用统计ipynb文件进行统计，存储为config_info.xlsx文件。
