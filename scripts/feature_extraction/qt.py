import numpy as np
import logging
from r_peak_detection import detect_r_peaks
from scipy.signal import find_peaks

# 创建独立的日志记录器
logger = logging.getLogger('qt_features')
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('../logging/qt_features.log')
file_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

def detect_qt_interval(ecg_signal, r_peaks, sampling_rate=512):
    """检测QT间期：通过Q波起点到T波终点"""
    qt_intervals = []
    for i in range(1, len(r_peaks) - 1):
        # 找到R波前后对应的Q波起点和T波终点
        q_peak_idx = find_q_peak(ecg_signal, r_peaks[i - 1], r_peaks[i])
        t_peak_idx = find_t_peak(ecg_signal, r_peaks[i], r_peaks[i + 1])
        
        if q_peak_idx is not None and t_peak_idx is not None:
            qt_interval = (t_peak_idx - q_peak_idx) / sampling_rate  # QT间期，单位为秒
            qt_intervals.append(qt_interval)
    return qt_intervals

def find_q_peak(ecg_signal, start_idx, end_idx):
    """根据R波位置找到Q波起点"""
    q_peak_idx = np.argmin(ecg_signal[start_idx:end_idx]) + start_idx
    return q_peak_idx

def find_t_peak(ecg_signal, start_idx, end_idx):
    """根据R波位置找到T波终点"""
    t_peak_idx = np.argmax(ecg_signal[start_idx:end_idx]) + start_idx
    return t_peak_idx

def calculate_qtc(qt_intervals, rr_intervals):
    """计算QTc（Bazett公式）"""
    qtc = [qt / np.sqrt(rr) for qt, rr in zip(qt_intervals, rr_intervals)]
    return np.mean(qtc), np.std(qtc)  # 返回平均QTc和QTc标准差

def calculate_qtd(qt_intervals):
    """计算QTd（QT间期离散度）"""
    # print("yeah")
    return np.std(qt_intervals)  # 返回QT间期的标准差作为QTd

def extract_qt_features(data_dict, sampling_rate=512):
    """从ECG数据字典中提取QT相关特征（QT间期、QTc、QTd）"""
    qt_features = []
    for subject, data_info in data_dict.items():
        data = data_info['data']
        r_peaks = detect_r_peaks(data, sampling_rate)
        
        # 计算QT间期
        qt_intervals = detect_qt_interval(data, r_peaks, sampling_rate)
        
        # 计算RR间期
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        # 计算QTc和QTd
        mean_qtc, std_qtc = calculate_qtc(qt_intervals, rr_intervals)
        qtd = calculate_qtd(qt_intervals)
        
        # 将QT特征添加到特征矩阵
        qt_features.append([mean_qtc, std_qtc, qtd])
    
    return np.array(qt_features)
