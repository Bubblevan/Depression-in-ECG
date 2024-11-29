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
from feature_extraction.hrv import extract_hr_hrv_features
from feature_extraction.qt import extract_qt_features
from feature_extraction.twa import extract_twa_features
from models.frets import BackboneFreTS
from models.fits import BackboneFITS

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score

def convert_to_tensor(X_train, X_test, y_train, y_test):
    # 将列表转换为 numpy 数组
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)
    y_train_array = np.array(y_train)
    y_test_array = np.array(y_test)
    
    # 将 numpy 数组转换为张量
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.long)

    # 确保张量形状为 (batch_size, n_features, n_steps)
    if X_train_tensor.ndim == 2:
        X_train_tensor = X_train_tensor.unsqueeze(1)  # 添加特征维度，假设单通道
    if X_test_tensor.ndim == 2:
        X_test_tensor = X_test_tensor.unsqueeze(1)    # 添加特征维度，假设单通道
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=1):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
    tn, fp, fn, tp = cm.ravel()

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 计算F1 score、Precision和Recall
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    return epoch_loss, epoch_accuracy, cm, fpr, tpr, roc_auc, f1, precision, recall
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)  # 确保模型加载到 CUDA 设备
    
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
        
        # 收集 ROC 曲线数据
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

    # 所有 epoch 结束后绘制 ROC 曲线
    plt.figure()
    for i in range(num_epochs):
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
    all_data = process_ecg_data2(base_path, sds_scale_path)

    # 进行基线修正
    all_data = correct_baseline(all_data, wavelet='db4', level=1)

    # 进行滤波
    all_data = filter_ecg_data(all_data)

    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = split_data(all_data)
    
    # 打印数据集大小
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print("训练集标签分布:", Counter(y_train))
    print("测试集标签分布:", Counter(y_test))
    # 转换为 PyTorch Tensor
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test)
    
    # 创建 DataLoader
    train_loader, test_loader = create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=1)  # 例如，减小batch_size以增加DataLoader大小
    
    # 打印 DataLoader 大小
    print(f"训练集 DataLoader 大小: {len(train_loader)}, 测试集 DataLoader 大小: {len(test_loader)}")
    
    # 模型参数设置
    n_steps = 100  # 根据您的数据调整
    n_features = 1  # 单通道ECG
    n_pred_steps = 10  # 根据您的任务调整
    cut_freq = 50  # 根据您的数据特性调整
    individual = False  # 如果是单通道，通常无需个体上采样
    num_classes = 2  # 二分类
    
    # 初始化模型
    model = BackboneFITS(n_steps, n_features, n_pred_steps, cut_freq, individual, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(f"Using device: {device}")
    model.to(device)

    # 计算类别权重以应对数据不平衡
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和评估
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=5)  # 可以增加epochs数量

if __name__ == "__main__":
    main()
