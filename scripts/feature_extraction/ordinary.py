import scipy.signal
import numpy as np
from .qrs import extract_wavelet_features

def extract_qrs_features(signal, sampling_rate=512):
    """
    提取QRS相关特征。
    返回一个字典，包含多个QRS特征。
    """
    # 使用scipy的find_peaks来检测QRS波
    # 这里仅为示例，实际应用中建议使用更精确的QRS检测算法
    peaks, _ = scipy.signal.find_peaks(signal, distance=sampling_rate*0.4)  # 假设最低心率为150 BPM，即间隔0.4秒
    qrs_amplitudes = signal[peaks]
    
    # 计算QRS相关特征
    if len(peaks) < 2:
        # 如果检测不到足够的QRS波，返回默认值
        qrs_features = {
            'qrs_amplitude_mean': 0,
            'qrs_amplitude_std': 0,
            'rr_interval_mean': 0,
            'rr_interval_std': 0
        }
    else:
        qrs_features = {
            'qrs_amplitude_mean': np.mean(qrs_amplitudes),
            'qrs_amplitude_std': np.std(qrs_amplitudes),
            'rr_interval_mean': np.mean(np.diff(peaks)) / sampling_rate,
            'rr_interval_std': np.std(np.diff(peaks)) / sampling_rate
        }
    return qrs_features

def extract_features(all_data, wavelet='db1', level=2, target_length=100, sampling_rate=512):
    """
    提取小波变换系数和QRS特征，并将其堆叠为多通道特征。
    每个信号将包含原始信号、小波系数和QRS特征。
    """
    all_features = []
    all_labels = []
    
    for subject, data_info in all_data.items():
        label = data_info['label']
        for signal in data_info['data']:
            # 确保信号长度为 target_length
            if len(signal) != target_length:
                signal = np.pad(signal, (0, max(0, target_length - len(signal))), 'constant')[:target_length]
            
            # 提取小波变换特征
            wavelet_coeffs = extract_wavelet_features(signal, wavelet=wavelet, level=level, target_length=target_length)  # shape: (num_coeffs, target_length)
            
            # 提取QRS特征
            qrs_feats = extract_qrs_features(signal, sampling_rate=sampling_rate)  # dict
            
            # 将QRS特征扩展为与时间步数相同的长度
            qrs_feature_array = np.array([qrs_feats['qrs_amplitude_mean'], 
                                          qrs_feats['qrs_amplitude_std'],
                                          qrs_feats['rr_interval_mean'],
                                          qrs_feats['rr_interval_std']])
            qrs_feature_expanded = np.tile(qrs_feature_array[:, np.newaxis], (1, target_length))  # shape: (4, target_length)
            
            # 堆叠所有特征作为多通道
            # 通道顺序：原始信号, cA2, cD2, cD1, QRS_mean, QRS_std, RR_mean, RR_std
            original_signal = signal[np.newaxis, :]  # shape: (1, target_length)
            stacked_features = np.vstack([original_signal, wavelet_coeffs, qrs_feature_expanded])  # shape: (1 + num_coeffs + 4, target_length)
            
            all_features.append(stacked_features)
            all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels)  # shapes: (num_samples, num_channels, target_length), (num_samples,)
