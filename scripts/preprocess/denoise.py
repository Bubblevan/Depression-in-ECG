import numpy as np
import pywt
import matplotlib.pyplot as plt

# 小波去噪函数
def wavelet_denoise(ecg_signal, wavelet='bior2.6', level=8):
    """
    小波去噪：去除高频噪声和基线漂移
    参数:
    - ecg_signal: ndarray, 原始ECG信号
    - wavelet: str, 小波类型
    - level: int, 小波分解层数
    返回:
    - denoised_signal: ndarray, 去噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # 处理细节系数，去除1,2层高频噪声
    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])

    # 处理近似系数，去除基线漂移
    coeffs[-1] = np.zeros_like(coeffs[-1])

    # 小波重构
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

def calculate_snr(original_signal, denoised_signal):
    """
    计算信噪比 (SNR)
    参数:
    - original_signal: ndarray, 原始信号
    - denoised_signal: ndarray, 去噪后的信号
    返回:
    - snr: float, 信噪比（分贝）
    """
    noise = original_signal - denoised_signal
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
