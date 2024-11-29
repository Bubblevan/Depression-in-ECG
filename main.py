import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import savgol_filter
import pywt

# 定义读取.bin文件的函数
def read_bindata(filepath, filename):
    fileFullPath = os.path.join(filepath, filename)
    with open(fileFullPath, 'rb') as fidin:
        dataTen = bytearray(fidin.read())
    data = []
    for n in range(int((len(dataTen) - 528 - 208) / 2)):
        value = dataTen[529 + 2 * n] + 256 * dataTen[529 + 2 * n + 1]
        data.append(value)
    return data

# 加载数据并打标签
def load_and_label_data(data_folder, label_file_path):
    df_labels = pd.read_excel(label_file_path)
    df_labels['ID'] = df_labels['ID'].astype(str).apply(lambda x: x.zfill(3))
    id_to_label = df_labels.set_index('ID')['Category'].to_dict()
    
    all_data = {}
    for subdir, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".bin"):
                src_file = os.path.join(subdir, file)
                subfolder = os.path.basename(subdir)
                data = read_bindata(subdir, file)
                file_id = subfolder.zfill(3)
                label = id_to_label.get(file_id, None)
                if label is not None:
                    all_data[file_id] = {'data': data, 'label': label}
                    print(f"Loaded data for {file_id} with label {label}")
                else:
                    print(f"No label found for {file_id}, skipping.")
    return all_data

# 滤波和基线校正函数
def smooth_and_correct_baseline(data):
    # Savitzky-Golay 滤波
    filtered_data = savgol_filter(data, window_length=11, polyorder=3)
    
    # 小波变换基线校正
    coeffs = pywt.wavedec(filtered_data, 'db4', level=1)
    coeffs[-1] *= 0  # 去除低频分量（基线）
    corrected_data = pywt.waverec(coeffs, 'db4')
    
    # 保证数据长度一致
    return corrected_data[:len(data)]

# 提取特征
def extract_features(signal, sampling_rate=512):
    # 信号预处理：滤波和基线修正
    preprocessed_signal = smooth_and_correct_baseline(signal)
    
    # R波检测
    cleaned_signal = nk.ecg_clean(preprocessed_signal, sampling_rate=sampling_rate, method="pantompkins")
    rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)[0]['ECG_R_Peaks']
    
    if len(rpeaks) < 2:
        return None  # 如果R波检测不到，返回None
    
    rr_intervals = np.diff(rpeaks) / sampling_rate
    if len(rr_intervals) == 0:
        return None
    
    # 计算心率和心率变异性
    heart_rate = 60 / np.mean(rr_intervals)
    hrv = np.std(rr_intervals)
    
    if np.isnan(heart_rate) or np.isnan(hrv):
        return None  # 如果计算的特征包含NaN，跳过该样本
    
    return heart_rate, hrv

# 定义简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # 设置路径
    data_folder = "./datasets/ECG_experiment"
    label_file_path = "./datasets/SDS_clean.xlsx"
    
    # 加载数据并打标签
    all_data = load_and_label_data(data_folder, label_file_path)
    
    # 提取特征和标签
    features = []
    labels = []
    for file_id, data_dict in all_data.items():
        feat = extract_features(data_dict['data'])
        if feat is None:
            print(f"特征提取失败，跳过 {file_id}")
            continue
        heart_rate, hrv = feat
        features.append([heart_rate, hrv])
        labels.append(data_dict['label'])
    
    # 转换为NumPy数组
    X = np.array(features)
    y = np.array(labels)
    
    # 检查并去除无效样本
    valid_indices = np.where(np.isfinite(X).all(axis=1))[0]
    X = X[valid_indices]
    y = y[valid_indices]
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 转换为PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 实例化模型、定义损失函数和优化器
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.cpu().numpy())
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predicted.cpu().numpy()))

if __name__ == '__main__':
    main()
