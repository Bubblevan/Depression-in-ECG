import numpy as np
from scipy.signal import savgol_filter
import logging

def apply_savitzky_golay_filter(ecg_signal, window_length=11, polyorder=3):
    """对单条ECG信号应用Savitzky-Golay滤波"""
    # 确保窗口长度为奇数且不超过信号长度
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(ecg_signal):
        window_length = len(ecg_signal) if len(ecg_signal) % 2 != 0 else len(ecg_signal) - 1
    return savgol_filter(ecg_signal, window_length=window_length, polyorder=polyorder)

def filter_ecg_data(data_dict):
    """对所有ECG数据应用Savitzky-Golay滤波"""
    filtered_data = {}
    for subject, data_info in data_dict.items():
        logging.info(f"Filtering ECG data for subject {subject}.")
        filtered_signals = []
        for ecg_signal in data_info['data']:
            filtered_signal = apply_savitzky_golay_filter(ecg_signal)
            filtered_signals.append(filtered_signal)
        filtered_data[subject] = {'data': filtered_signals, 'label': data_info['label']}
    return filtered_data
