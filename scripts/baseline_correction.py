import pywt
import numpy as np
import logging

def correct_baseline(data, wavelet='db4', level=1):
    """使用小波变换进行ECG基线漂移校正"""
    corrected_data_dict = {}
    for subject, data_info in data.items():
        logging.info(f"Processing baseline correction for subject {subject}.")
        corrected_signals = []
        for ecg_signal in data_info['data']:
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
            coeffs[-1] = np.zeros_like(coeffs[-1])  # 将基线的近似系数设置为0
            corrected_signal = pywt.waverec(coeffs, wavelet)
            # 确保校正后的数据长度与原始数据相同
            corrected_signal = corrected_signal[:len(ecg_signal)]
            corrected_signals.append(corrected_signal)
        corrected_data_dict[subject] = {'data': corrected_signals, 'label': data_info['label']}
    return corrected_data_dict
