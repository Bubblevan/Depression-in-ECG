import numpy as np
import pywt
import matplotlib.pyplot as plt

# С��ȥ�뺯��
def wavelet_denoise(ecg_signal, wavelet='bior2.6', level=8):
    """
    С��ȥ�룺ȥ����Ƶ�����ͻ���Ư��
    ����:
    - ecg_signal: ndarray, ԭʼECG�ź�
    - wavelet: str, С������
    - level: int, С���ֽ����
    ����:
    - denoised_signal: ndarray, ȥ�����ź�
    """
    # С���ֽ�
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # ����ϸ��ϵ����ȥ��1,2���Ƶ����
    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])

    # �������ϵ����ȥ������Ư��
    coeffs[-1] = np.zeros_like(coeffs[-1])

    # С���ع�
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

def calculate_snr(original_signal, denoised_signal):
    """
    ��������� (SNR)
    ����:
    - original_signal: ndarray, ԭʼ�ź�
    - denoised_signal: ndarray, ȥ�����ź�
    ����:
    - snr: float, ����ȣ��ֱ���
    """
    noise = original_signal - denoised_signal
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
