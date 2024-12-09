import numpy as np
import logging
from baseline_correction import correct_baseline
from feature_extraction.r_peak_detection import detect_r_peaks

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
        logger.warning("R-peaks less than 2, cannot calculate HR and HRV.")
        return None, None

    # 计算 RR 间期
    rr_intervals = np.diff(r_peaks) / sampling_rate  # RR间期，单位为秒
    if len(rr_intervals) == 0 or np.any(rr_intervals <= 0):
        logger.error("Invalid RR intervals detected. Check R-peak detection.")
        return None, None

    # 过滤异常 RR 间期（基于生理范围：0.4s - 2.0s）
    rr_intervals_filtered = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 2.0)]
    if len(rr_intervals_filtered) < 2:
        logger.error("Insufficient valid RR intervals after filtering.")
        return None, None

    # 计算心率和 HRV
    heart_rate = 60 / rr_intervals_filtered  # 心率，单位为 bpm
    heart_rate_variability = np.std(rr_intervals_filtered)  # HRV
    return np.mean(heart_rate), heart_rate_variability

def extract_hr_hrv_features(data_dict, sampling_rate=512):
    """从 ECG 数据字典中提取心率和心率变异性特征"""
    features_matrix = []
    for subject, data_info in data_dict.items():
        logger.info(f"Processing data for subject {subject}.")
        data = data_info['data']

        # 检测 R 波
        r_peaks = detect_r_peaks(data, sampling_rate)

        # 计算心率和心率变异性
        heart_rate, heart_rate_variability = calculate_hr_hrv(r_peaks, sampling_rate)

        # 如果计算结果有效，记录日志
        if heart_rate is not None and heart_rate_variability is not None:
            logger.info(f"Subject {subject}: Heart Rate: {heart_rate:.2f} bpm, HRV: {heart_rate_variability:.2f} seconds")
            features_matrix.append([heart_rate, heart_rate_variability])
        else:
            logger.warning(f"Subject {subject}: Invalid data, skipping this subject.")
    
    return np.array(features_matrix)

