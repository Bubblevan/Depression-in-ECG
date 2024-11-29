import scipy.io
import matplotlib.pyplot as plt

# 读取 .mat 文件
mat_file_path = '../datasets/depression_recognition/train set/001-1.mat'
mat_data = scipy.io.loadmat(mat_file_path)


ecg_signal = mat_data['segmentData']

# 信号是二维的
ecg_signal = ecg_signal[0]  # 选择第一个通道

# 截取前面几条数据
num_samples_to_plot = 1000 
ecg_signal_subset = ecg_signal[:num_samples_to_plot]

# 生成时间轴（假设采样率为 1 Hz）
time = range(num_samples_to_plot)

plt.figure(figsize=(10, 6))
plt.plot(time, ecg_signal_subset, label='ECG Signal', drawstyle='default')
plt.title('ECG Signal (First {} Samples)'.format(num_samples_to_plot))
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()