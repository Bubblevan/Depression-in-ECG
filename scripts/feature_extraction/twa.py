import numpy as np
import logging
from scipy.signal import find_peaks
from r_peak_detection import detect_r_peaks

# 创建独立的日志记录器
logger = logging.getLogger('twa_features')
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('../logging/twa_features.log')
file_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

def detect_twa(ecg_signal, r_peaks, sampling_rate=512):
    """检测T波电交替（TWA）特征"""
    twa_features = []
    for i in range(1, len(r_peaks) - 1):
        # 提取每个R-R间期内的T波（R波后，取一定窗口的范围）
        start_idx = r_peaks[i] + int(0.2 * sampling_rate)  # 假设T波起始在R波之后200ms
        end_idx = r_peaks[i] + int(0.6 * sampling_rate)  # 假设T波结束在R波之后600ms

        if end_idx < len(ecg_signal):
            t_wave_segment = ecg_signal[start_idx:end_idx]
            
            # 计算T波振幅
            twa_amplitude = np.max(t_wave_segment) - np.min(t_wave_segment)
            twa_features.append(twa_amplitude)

    return np.array(twa_features)

def extract_twa_features(data_dict, sampling_rate=512):
    """从ECG数据字典中提取TWA特征"""
    twa_features = []
    for subject, data_info in data_dict.items():
        data = data_info['data']
        r_peaks = detect_r_peaks(data, sampling_rate)
        
        # 检测TWA特征
        twa_amplitudes = detect_twa(data, r_peaks, sampling_rate)
        
        # 计算TWA的均值和标准差
        mean_twa = np.mean(twa_amplitudes)
        std_twa = np.std(twa_amplitudes)
        
        # 将TWA特征添加到特征矩阵
        twa_features.append([mean_twa, std_twa])
    
    return np.array(twa_features)
