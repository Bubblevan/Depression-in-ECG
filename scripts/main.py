import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from collections import Counter

from data_process import process_ecg_data1, process_ecg_data2, split_data
from ecg_filtering import filter_ecg_data
from baseline_correction import correct_baseline
from feature_extraction.ordinary import extract_features 
from feature_extraction.hrv import extract_hr_hrv_features
from feature_extraction.qt import extract_qt_features
from feature_extraction.twa import extract_twa_features
from models.frets import BackboneFreTS
from models.fits import BackboneFITS
from models.wptcn import WPTCN

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def convert_to_tensor(X, y):
    """
    将特征和标签转换为PyTorch张量。
    """
    # 将列表转换为 numpy 数组
    X_array = np.array(X)
    y_array = np.array(y)
    
    # 将 numpy 数组转换为张量
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.long)

    return X_tensor, y_tensor

def create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=16):
    """
    创建训练集和测试集的DataLoader。
    """
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个epoch。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

def evaluate(model, test_loader, criterion, device):
    """
    评估模型性能。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = 100 * correct / total

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 计算F1 score、Precision和Recall
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions)

    return epoch_loss, epoch_accuracy, cm, fpr, tpr, roc_auc, f1, precision, recall

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=20):
    """
    训练并评估模型。
    """
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_auc = 0
    early_stop_patience = 10
    epochs_no_improve = 0
    all_fpr = []
    all_tpr = []
    all_roc_auc = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        test_loss, test_accuracy, cm, fpr, tpr, roc_auc, f1, precision, recall = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Confusion Matrix:\n{cm}")
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("-" * 40)

        # 调整学习率
        scheduler.step(test_loss)

        # 早停法
        if roc_auc > best_auc:
            best_auc = roc_auc
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

        # 收集 ROC 曲线数据
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

    # 绘制 ROC 曲线
    plt.figure()
    for i in range(len(all_fpr)):
        plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'ROC curve (area = {all_roc_auc[i]:.2f}) epoch {i+1}')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def main():
    # 合并正负类数据
    # positive_data = process_ecg_data1(base_path_positive, label=1)
    # negative_data = process_ecg_data1(base_path_negative, label=0)
    # all_data = {**positive_data, **negative_data}
    # 基线校正
    # corrected_data = correct_baseline(all_data)
        
    # 滤波处理
    # filtered_data = filter_ecg_data(corrected_data)
    
    # 提取心率和心率变异性特征
    # hrv_features = extract_hr_hrv_features(filtered_data, sampling_rate)
    
    # 提取QT间期相关特征
    # qt_features = extract_qt_features(filtered_data, sampling_rate)
    
    # 提取TWA特征
    # twa_features = extract_twa_features(filtered_data, sampling_rate)
    
    # 合并所有特征
    # all_features = np.hstack((hrv_features, qt_features, twa_features))
    
    # 输出结果
    # print("特征矩阵形状:", all_features.shape)
    # print("特征矩阵示例:", all_features[:5])  # 显示前5个样本
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设置路径
    base_path = '../datasets/depression_recognition'
    sds_scale_path = os.path.join(base_path, 'SDS scale.xlsx')

    # 处理数据
    all_data = process_ecg_data2(base_path, sds_scale_path, target_length=100)

    if all_data is None:
        print("数据处理过程中存在标签不一致的问题，停止训练。")
        return

    # 进行基线修正
    all_data = correct_baseline(all_data, wavelet='db4', level=1)

    # 进行滤波
    all_data = filter_ecg_data(all_data)

    # 提取特征
    X, y = extract_features(all_data, wavelet='db1', level=2, target_length=10000, sampling_rate=512)

    # 打印特征和标签形状
    print(f"特征矩阵形状: {X.shape}, 标签向量形状: {y.shape}")

    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 打印数据集大小和标签分布
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print("训练集标签分布:", Counter(y_train))
    print("测试集标签分布:", Counter(y_test))

    # 转换为 PyTorch Tensor
    X_train_tensor, y_train_tensor = convert_to_tensor(X_train, y_train)
    X_test_tensor, y_test_tensor = convert_to_tensor(X_test, y_test)

    # 创建 DataLoader
    train_loader, test_loader = create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=16)  # 使用较大的 batch_size

    # 打印 DataLoader 大小
    print(f"训练集 DataLoader 大小: {len(train_loader)}, 测试集 DataLoader 大小: {len(test_loader)}")

    # 模型参数设置
    num_input_channels = X.shape[1]  # 特征通道数，根据extract_features函数确定
    input_length = X.shape[2]        # 时间步数
    num_classes = 2                   # 二分类
    hidden_dim = 64                   # 隐藏层维度，可根据需求调整
    kernel_size = 4                   # 卷积核大小
    num_levels = 2                    # 小波包分解层数
    num_layers = 3                    # 模型深度（WTTCNBlock的数量）
    wavelet_type = 'db1'              # 小波类型
    feedforward_ratio = 2             # 前馈网络扩展比例
    group_type = 'channel'            # 分组类型
    normalization_eps = 1e-5         # 归一化epsilon
    normalization_affine = True       # 归一化是否有仿射变换

    # 初始化模型
    model = WPTCN(
        num_input_channels=num_input_channels,
        input_length=input_length,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_levels=num_levels,
        num_layers=num_layers,
        wavelet_type=wavelet_type,
        feedforward_ratio=feedforward_ratio,
        group_type=group_type,
        normalization_eps=normalization_eps,
        normalization_affine=normalization_affine
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(f"Using device: {device}")
    model.to(device)

    # 计算类别权重以应对数据不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和评估
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=20)  # 增加epochs数量

if __name__ == "__main__":
    main()