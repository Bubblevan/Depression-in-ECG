import pywt
import numpy as np

def extract_wavelet_features(signal, wavelet='db1', level=2, target_length=100):
    """
    提取小波变换系数并调整长度到 target_length。
    返回多个通道的特征数组。
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs = [cA2, cD2, cD1] for level=2
    processed_coeffs = []
    for coeff in coeffs:
        current_length = len(coeff)
        if current_length < target_length:
            # 上采样
            coeff_interp = np.interp(
                np.linspace(0, current_length, target_length),
                np.arange(current_length),
                coeff
            )
        elif current_length > target_length:
            # 下采样
            step = current_length / target_length
            indices = (np.arange(target_length) * step).astype(int)
            coeff_interp = coeff[indices]
        else:
            coeff_interp = coeff
        processed_coeffs.append(coeff_interp)
    return np.array(processed_coeffs)  # shape: (num_coeffs, target_length)