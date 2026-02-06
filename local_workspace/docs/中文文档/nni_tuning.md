# nni 迭代

进行参数优化等相关操作

## 文件结构

C:\USERS\LUMISONG\DESKTOP\插补库\SAITS\NNI_TUNING
├─BRITS
│      BRITS_basic_config.ini
│      BRITS_searching_config.yml
│      BRITS_searching_space.json
│
├─MRNN
│      MRNN_basic_config.ini
│      MRNN_searching_config.yml
│      MRNN_searching_space.json
│
├─SAITS
│      SAITS_basic_config.ini
│      SAITS_searching_config.yml
│      SAITS_searching_space.json
│
└─Transformer
        Transformer_basic_config.ini
        Transformer_searching_config.yml
        Transformer_searching_space.json

含不同机器学习模型的配置和搜索空间定义文件：BRITS、MRNN、SAITS 和 Transformer。

1. model_basic_config.ini ：这些可能是每个模型的基本配置文件，指定模型的默认或初始参数。
2. model_searching_config.yml ：这些文件可能用于超参数调整或模型选择。它们可能包含搜索算法的设置，例如试验次数或搜索方法的类型。
3. model_searching_space.json ：这些文件可能定义超参数的搜索空间。它们指定要调整的超参数，以及每个超参数可以采用的范围或值集。

这种目录结构和文件类型表明该项目涉及不同类型模型的超参数调整，可能使用 Microsoft 的神经网络智能 (NNI) 工具包，该工具包是一种用于自动化机器学习实验的工具。
