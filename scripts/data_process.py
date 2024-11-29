import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# def read_bindata(file_path):
#     """读取.bin文件并返回ECG数据"""
#     with open(file_path, 'rb') as fidin:
#         # 跳过文件头部的非数据部分，根据MATLAB代码，数据从第529个字节开始
#         fidin.seek(528)
#         # 读取数据部分
#         data_raw = fidin.read()
    
#     # 根据MATLAB代码，每个数据点由两个字节组成
#     data = np.frombuffer(data_raw, dtype=np.uint16)
    
#     # 截取前500000个数据点
#     data_500000 = data[:500000]
    
#     return data_500000
def read_bindata(filepath):
    with open(filepath, 'rb') as fidin:
        dataTen = bytearray(fidin.read())
    data = []
    for n in range(int((len(dataTen) - 528 - 208) / 2)):
        value = dataTen[529 + 2 * n] + 256 * dataTen[529 + 2 * n + 1]
        data.append(value)
    return data

def process_ecg_data1(base_path, label):
    """处理ECG数据并返回数据字典，同时添加标签"""
    all_data = {}
    # 遍历所有被试文件夹
    for subject in range(1, 90):  # 被试编号从001到089
        subject_folder = f"{subject:03d}"  # 格式化编号为三位数，例如001
        subject_path = os.path.join(base_path, subject_folder)
        
        # 检查被试文件夹是否存在
        if not os.path.isdir(subject_path):
            # print(f"被试文件夹 {subject_folder} 不存在，跳过。")
            continue
        
        # 获取.bin文件路径
        bin_files = [f for f in os.listdir(subject_path) if f.endswith('.bin')]
        if not bin_files:
            print(f"被试 {subject_folder} 中没有.bin文件，跳过。")
            continue
        
        bin_file_path = os.path.join(subject_path, bin_files[0])
        
        # 读取并处理数据
        ecg_data = read_bindata(bin_file_path)
        
        # 保存到字典中，并添加标签
        all_data[subject_folder] = {'data': ecg_data, 'label': label}
    
    return all_data



######################################################################################
def read_matdata(filepath):
    """读取 .mat 文件中的 ECG 信号数据"""
    # 检查文件名是否符合格式 001-1.mat
    filename = os.path.basename(filepath)
    if not filename.startswith(tuple([f"{i:03d}" for i in range(1, 47)])) or '-' not in filename:
        print(f"文件名 {filename} 不符合格式，跳过。")
        return None
    
    mat_data = scipy.io.loadmat(filepath)
    ecg_signal = mat_data['segmentData']
    return ecg_signal[0]  # 信号是二维的，选择第一个通道

def process_ecg_data2(base_path, sds_scale_path, target_length=100):
    """处理 ECG 数据并返回数据字典，同时添加标签"""
    all_data = {}
    
    # 读取 SDS scale 文件
    sds_scale = pd.read_excel(sds_scale_path)
    
    # 遍历 train set 和 test set 文件夹
    for dataset_type in ['train set', 'test set']:
        dataset_path = os.path.join(base_path, dataset_type)
        
        if not os.path.exists(dataset_path):
            logging.warning(f"路径 {dataset_path} 不存在，跳过。")
            continue
        
        # 获取 .mat 文件路径
        mat_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_path) for file in files if file.endswith('.mat')]
        
        if not mat_files:
            logging.warning(f"数据集 {dataset_type} 中没有 .mat 文件，跳过。")
            continue
        
        # 读取并处理所有 .mat 文件
        for mat_file_path in mat_files:
            ecg_data = read_matdata(mat_file_path)
            if ecg_data is not None:
                # 从文件名中提取被试编号
                filename = os.path.basename(mat_file_path)
                subject_folder = filename.split('-')[0]
                
                # 获取标签
                try:
                    label = sds_scale[sds_scale.iloc[:, 0] == int(subject_folder)].iloc[:, -1].values[0]
                except IndexError:
                    logging.error(f"被试编号 {subject_folder} 在SDS scale中未找到标签，跳过。")
                    continue
                
                # 填充或截断ECG信号到目标长度
                if len(ecg_data) < target_length:
                    padding = target_length - len(ecg_data)
                    ecg_data = np.pad(ecg_data, (0, padding), 'constant')
                elif len(ecg_data) > target_length:
                    ecg_data = ecg_data[:target_length]
                
                # 保存到字典中，并添加标签
                if subject_folder not in all_data:
                    all_data[subject_folder] = {'data': [], 'label': label}
                all_data[subject_folder]['data'].append(ecg_data)
    
    return all_data

def split_data(all_data, test_size=0.2, random_state=42):
    # Separate subjects by label
    positive_subjects = []
    negative_subjects = []
    for subject, data in all_data.items():
        if data['label'] == 1:
            positive_subjects.append(subject)
        elif data['label'] == 0:
            negative_subjects.append(subject)
        else:
            print(f"Subject {subject} has invalid label: {data['label']}")

    # Split positive and negative subjects
    pos_train, pos_test = train_test_split(positive_subjects, test_size=test_size, random_state=random_state)
    neg_train, neg_test = train_test_split(negative_subjects, test_size=test_size, random_state=random_state)

    # Combine train and test subjects
    train_subjects = pos_train + neg_train
    test_subjects = pos_test + neg_test

    # Collect data and labels
    X_train = []
    y_train = []
    for subj in train_subjects:
        X_train.extend(all_data[subj]['data'])
        y_train.extend([all_data[subj]['label']] * len(all_data[subj]['data']))

    X_test = []
    y_test = []
    for subj in test_subjects:
        X_test.extend(all_data[subj]['data'])
        y_test.extend([all_data[subj]['label']] * len(all_data[subj]['data']))

    return X_train, X_test, y_train, y_test