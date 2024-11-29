# 项目概述
本项目旨在处理心电图（ECG）数据，提取心率变异性（HRV）特征，并使用2024年最新的时序模型对ECG数据进行端到端的训练。项目包含两个主要数据集：ECSMP和课设数据集，均通过SDS（Self-Rating Depression Scale）进行标记。
# 目录结构
```
.
├── main.py
├── scripts
│   └── main.py
│   └──feature_extraction
│       └──hrv/qt/twa
├── datasets
│   ├── ECG_experiment
│   └── depression_recognition
└── models
```
# 文件说明
`main.py`
功能：提取ECSMP数据集的心率变异性（HRV）特征。
使用方法：运行该脚本以提取ECSMP数据集的HRV特征。

`scripts/main.py`
功能：对课设ECG数据集进行端到端的时序模型训练。
使用方法：运行该脚本以直接对课设ECG数据集进行时序模型训练。

datasets/ECG_experiment：ECSMP数据集，包含用于提取HRV特征的ECG数据。

depression_recognition：课设数据集，包含通过SDS标记的ECG数据，用于时序模型训练。

models: 功能：包含2024年最新的时序模型，用于对ECG数据进行训练和预测。

# 后续规划
传统机器学习？加特征？排除bug？