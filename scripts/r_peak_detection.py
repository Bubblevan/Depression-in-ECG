import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(ecg_signal, sampling_rate=512):
    """使用Pan-Tompkins算法检测ECG信号中的R波"""
    diff_signal = np.diff(ecg_signal)
    squared_signal = diff_signal ** 2
    
    # 调试：打印 squared_signal 的形状和类型
    # print(f"squared_signal shape: {squared_signal.shape}, dtype: {squared_signal.dtype}")
    
    # 确保 squared_signal 是一维数组
    if squared_signal.ndim != 1:
        raise ValueError(f"Expected squared_signal to be a 1D array, but got {squared_signal.ndim}D array.")
    
    integrated_signal = np.convolve(squared_signal, np.ones(30), mode='same')
    
    peaks, _ = find_peaks(integrated_signal, distance=sampling_rate * 0.3, height=0.1 * np.max(integrated_signal))
    return peaks

