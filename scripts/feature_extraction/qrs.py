import numpy as np
import pywt
import matplotlib.pyplot as plt


def wavelet_transform(ecg_signal, wavelet='bior2.6', level=4):
    """
    对 ECG 信号进行小波分解，提取指定层数的细节系数。
    参数:
    - ecg_signal: ndarray, 原始 ECG 信号
    - wavelet: str, 小波类型，默认 'bior2.6'
    - level: int, 小波分解的层数，默认 4
    返回:
    - coeffs: list, 小波分解的细节系数和近似系数
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    return coeffs


def detect_extrema(detail_coeffs):
    """
    检测细节系数中的极大值和极小值。
    参数:
    - detail_coeffs: ndarray, 第 3 层细节系数
    返回:
    - maxima: ndarray, 极大值位置
    - minima: ndarray, 极小值位置
    """
    maxima = (np.diff(np.sign(np.diff(detail_coeffs))) < 0).nonzero()[0] + 1  # 极大值
    minima = (np.diff(np.sign(np.diff(detail_coeffs))) > 0).nonzero()[0] + 1  # 极小值
    return maxima, minima


def detect_r_peaks(detail_coeffs, threshold_ratio=1/3):
    """
    检测 R 波位置。
    参数:
    - detail_coeffs: ndarray, 第 3 层细节系数
    - threshold_ratio: float, 用于设定阈值的比例，默认 1/3
    返回:
    - r_peaks: list, 检测到的 R 波位置
    """
    maxima, minima = detect_extrema(detail_coeffs)

    # 设置阈值
    threshold = threshold_ratio * np.mean([np.max(detail_coeffs[:len(detail_coeffs)//4]),
                                           np.max(detail_coeffs[len(detail_coeffs)//4:len(detail_coeffs)//2]),
                                           np.max(detail_coeffs[len(detail_coeffs)//2:3*len(detail_coeffs)//4]),
                                           np.max(detail_coeffs[3*len(detail_coeffs)//4:])])

    # 筛选极大值和极小值
    valid_maxima = maxima[detail_coeffs[maxima] > threshold]
    valid_minima = minima[detail_coeffs[minima] < -threshold]

    # 匹配极大值和极小值，寻找 R 波
    r_peaks = []
    for max_idx in valid_maxima:
        min_idx_candidates = valid_minima[valid_minima > max_idx]
        if len(min_idx_candidates) > 0:
            min_idx = min_idx_candidates[0]
            r_peak = (max_idx + min_idx) // 2
            r_peaks.append(r_peak)
    
    return r_peaks


def compensate_r_peaks(r_peaks, shift=10):
    """
    补偿 R 波位置的漂移。
    参数:
    - r_peaks: list, 检测到的 R 波位置
    - shift: int, 漂移补偿量，默认 10
    返回:
    - compensated_r_peaks: list, 补偿后的 R 波位置
    """
    return [r - shift for r in r_peaks]


def plot_ecg_with_r_peaks(ecg_signal, r_peaks, title="ECG Signal with R Peaks"):
    """
    绘制 ECG 信号并标注 R 波位置。
    参数:
    - ecg_signal: ndarray, ECG 信号
    - r_peaks: list, R 波位置
    """
    plt.figure(figsize=(12, 6))
    plt.plot(ecg_signal, label="ECG Signal")
    plt.scatter(r_peaks, ecg_signal[r_peaks], color='red', label="R Peaks")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
