load('013-2.mat');

signal_length = length(segmentData);
disp(['信号长度: ', num2str(signal_length)]);

% 查看采样率
if exist('Fs', 'var')
    disp(['采样率: ', num2str(Fs), ' Hz']);
elseif exist('time', 'var')
    dt = mean(diff(time));  % 计算时间步长
    Fs = 1 / dt;  % 计算采样率
    disp(['采样率: ', num2str(Fs), ' Hz']);
else
    disp('采样率信息未找到。');
end