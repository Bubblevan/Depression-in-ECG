import numpy as np
import logging
from feature_extraction.r_peak_detection import detect_r_peaks
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
    """因为Q 波总是局部最小值，并且在 R 波之前，所以在当前 R 波和前一个 R 波之间，找到信号的最小值"""
    q_peak_idx = np.argmin(ecg_signal[start_idx:end_idx]) + start_idx
    return q_peak_idx

def find_t_peak(ecg_signal, start_idx, end_idx):
    """T 波总是局部最大值，并且在 R 波之后，故在当前 R 波和下一个 R 波之间，找到信号的最大值"""
    t_peak_idx = np.argmax(ecg_signal[start_idx:end_idx]) + start_idx
    return t_peak_idx

def calculate_qtc(qt_intervals, rr_intervals):
    """计算QTc（Bazett公式）"""
    valid_intervals = [(qt, rr) for qt, rr in zip(qt_intervals, rr_intervals) if qt is not None and rr > 0]
    if not valid_intervals:  # 如果没有有效数据，返回默认值或抛出异常
        return None, None
    qtc = [qt / np.sqrt(rr) for qt, rr in valid_intervals]
    return np.mean(qtc), np.std(qtc)


def calculate_qtd(qt_intervals):
    """计算QTd（QT间期离散度）"""
    # print("yeah")
    return np.std(qt_intervals)  # 返回QT间期的标准差作为QTd


def extract_qt_features(ecg_signal, sampling_rate=512):
    """
    从单个 ECG 信号中提取 QT 特征，包括 QTc 和 QTd。

    参数:
    - ecg_signal: ndarray, 输入的 ECG 信号数据
    - sampling_rate: int, 采样率，默认为 512 Hz

    返回:
    - (mean_qtc, std_qtc, qtd): tuple, 平均 QTc，QTc 的标准差，QTd
    """
    # 检测 R 波
    r_peaks = detect_r_peaks(ecg_signal, sampling_rate)
    if len(r_peaks) < 3:
        print("Insufficient R-peaks detected. Skipping signal.")
        return None

    # 计算 QT 间期
    qt_intervals = detect_qt_interval(ecg_signal, r_peaks, sampling_rate)
    # qt_intervals = detect_qt_wavelets(ecg_signal, sampling_rate)
    if len(qt_intervals) == 0:
        print("No valid QT intervals detected.")
        return None

    # 计算 RR 间期
    rr_intervals = np.diff(r_peaks) / sampling_rate
    if len(rr_intervals) < len(qt_intervals):
        rr_intervals = rr_intervals[:len(qt_intervals)]  # 确保 RR 间期与 QT 间期数量一致

    # 计算 QTc 和 QTd
    mean_qtc, std_qtc = calculate_qtc(qt_intervals, rr_intervals)
    qtd = calculate_qtd(qt_intervals)

    return mean_qtc, std_qtc, qtd, qt_intervals, rr_intervals


def detect_qt_wavelets(ecg_signal, sampling_rate=512):
    """
    利用小波变换提取 Q 波起点和 T 波终点，并计算 QT 间期
    参数:
    - ecg_signal: ndarray, 输入的 ECG 信号
    - sampling_rate: int, 采样率（Hz），默认为 512

    返回:
    - qt_interval: float, QT 间期（单位：秒），如果无法计算则返回 None
    - q_peak_idx: int, Q 波起点索引（用于调试）
    - t_peak_idx: int, T 波终点索引（用于调试）
    """
    import pywt
    from scipy.signal import find_peaks

    # 小波变换
    coeffs = pywt.wavedec(ecg_signal, 'sym4', level=5)

    # 使用第 3 和 4 层系数（可调整）
    cD3 = coeffs[3]
    cD4 = coeffs[4]

    # 结合多个层系数特征，寻找极值
    q_peaks, _ = find_peaks(-np.abs(cD4), distance=int(sampling_rate * 0.1))  # Q 波近似位置
    t_peaks, _ = find_peaks(np.abs(cD3), distance=int(sampling_rate * 0.1))  # T 波近似位置

    # 检查是否检测到 Q 波和 T 波
    if len(q_peaks) == 0 or len(t_peaks) == 0:
        return None, None, None  # 无法检测到 QT 间期

    # 选择第一个检测到的 Q 波和 T 波
    q_peak_idx = q_peaks[0]
    t_peak_idx = t_peaks[0]

    # 确保 T 波在 Q 波之后
    if t_peak_idx <= q_peak_idx:
        return None, q_peak_idx, t_peak_idx  # 不合理的 QT 间期

    # 计算 QT 间期
    qt_interval = (t_peak_idx - q_peak_idx) / sampling_rate

    return qt_interval, q_peak_idx, t_peak_idx
