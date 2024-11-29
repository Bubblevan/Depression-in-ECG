import numpy as np
import logging
from baseline_correction import correct_baseline
from r_peak_detection import detect_r_peaks

# 创建独立的日志记录器
logger = logging.getLogger('hrv_features')
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('../logging/hrv_features.log')
file_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)

def calculate_hr_hrv(r_peaks, sampling_rate=512):
    """计算心率和心率变异性"""
    if len(r_peaks) < 2:
        logging.warning("R-peaks less than 2, cannot calculate HR and HRV.")
        return None, None
    
    rr_intervals = np.diff(r_peaks) / sampling_rate  # RR间期，单位为秒
    heart_rate = 60 / rr_intervals  # 心率，单位为bpm
    heart_rate_variability = np.std(rr_intervals)  # 心率变异性
    return np.mean(heart_rate), heart_rate_variability

def extract_hr_hrv_features(data_dict, sampling_rate=512):
    """从ECG数据字典中提取心率和心率变异性特征"""
    features_matrix = []
    for subject, data_info in data_dict.items():
        logging.info(f"Processing data for subject {subject}.")
        data = data_info['data']
        
 
        # R波检测
        r_peaks = detect_r_peaks(data, sampling_rate)
        
        # 计算心率和心率变异性
        heart_rate, heart_rate_variability = calculate_hr_hrv(r_peaks, sampling_rate)
        
        # 如果计算结果有效，记录日志
        if heart_rate is not None and heart_rate_variability is not None:
            logging.info(f"Subject {subject}: Heart Rate: {heart_rate:.2f} bpm, Heart Rate Variability: {heart_rate_variability:.2f} seconds")
            features_matrix.append([heart_rate, heart_rate_variability])
        else:
            logging.warning(f"Subject {subject}: Invalid data, skipping this subject.")
    
    return np.array(features_matrix)

