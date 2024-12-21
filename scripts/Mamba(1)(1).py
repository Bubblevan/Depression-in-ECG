# -*- coding: utf-8 -*-
import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import random_split


# 1. 加载.mat文件函数
def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    signal = data['signal_resized'].squeeze()  # 获取信号数据
    label = data['label'].squeeze()  # 获取标签
    return signal, label

# 2. 数据预处理：将多个信号数据和标签加载为numpy数组
def load_data(data_dir, start_idx, end_idx):
    signals = []
    labels = []
    for i in range(start_idx, end_idx + 1):  # 数据集编号从start_idx到end_idx
        for label in [1, 2]:  # 对应1或2的文件
            file_name = f'{i:03d}-{label}_processed.mat'
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                signal, label = load_mat_file(file_path)
                signals.append(signal)
                labels.append(label)
    return np.array(signals), np.array(labels)

# 3. 数据标准化
def preprocess_data(signals):
    scaler = StandardScaler()
    signals = np.array([scaler.fit_transform(signal.reshape(-1, 1)).flatten() for signal in signals])
    return signals

# 4. 定义PyTorch数据集
class SignalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

# 5. 创建卷积神经网络（CNN）模型
class CNNModel(nn.Module):
    def __init__(self, input_length):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        self._to_linear = self._get_conv_output(input_length)
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _get_conv_output(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        return x.numel()
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()

# 6. 主训练函数
def train_model(train_data_dir, val_data_dir, device, patience=10):
    X, y = load_data(train_data_dir, 1, 37)  # 训练集数据（1-37）
    X = preprocess_data(X)
    y = y.astype(np.float32)

    dataset = SignalDataset(X, y)
    train_size = int(0.7 * len(dataset))  # 70%用于训练
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_length = X.shape[1]
    model = CNNModel(input_length).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.05)

    epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * signals.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / train_size
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    # 绘制Loss曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

        # 绘制Accuracy曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    print("Visualization Complete!")

# 在train_model函数的最后调用plot_metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    return model

# 7. 测试模型函数
def test_model(test_data_dir, model, device):
    X, y = load_data(test_data_dir, 38, 46)  # 测试集数据（38-46）
    X = preprocess_data(X)
    y = y.astype(np.float32)

    dataset = SignalDataset(X, y)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * signals.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(dataset)
    test_accuracy = correct / total * 100
    print(f'Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 添加可视化调用
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    return avg_test_loss, test_accuracy

if __name__ == "__main__":
    # 检查是否有可用的GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 指定训练集和测试集路径
    train_data_dir = r"C:\Coding\BSP\Train\Processed_Data"
    val_data_dir = r"C:\Coding\BSP\Val\Processed_Data"  # 新增验证集路径
    test_data_dir = r"C:\Coding\BSP\Test\Processed_Data"

    # 训练模型，设置早停的耐心值为10
    model = train_model(train_data_dir, val_data_dir, device, patience=10)

    # 测试模型
    test_loss, test_accuracy = test_model(test_data_dir, model, device)