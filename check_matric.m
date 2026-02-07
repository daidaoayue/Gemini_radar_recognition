%% 矩阵可视化检查工具
clear; clc; close all;

% 1. 选择要查看的 .mat 文件
[file, path] = uigetfile('*.mat', '选择一个生成的 .mat 数据文件');
if isequal(file, 0)
    disp('用户取消了选择');
    return;
end

% 2. 加载数据
filepath = fullfile(path, file);
load(filepath); % 加载后工作区通常会出现 'data' 变量

% 检查变量是否存在
if ~exist('data', 'var')
    error('文件中未找到名为 "data" 的变量，请检查保存格式。');
end

% 3. 自动识别维度并绘图
% 如果是 3 通道数据 [3, 32, 64]，则分块显示
% 如果是单通道数据 [32, 64]，则直接显示
dim = size(data);

figure('Name', ['查看矩阵: ', file], 'Color', 'w', 'Position', [200, 200, 1000, 600]);

if length(dim) == 3 && dim(1) == 3
    % 情况 A: 三通道数据 (例如：1-RD图, 2-距离热力图, 3-速度热力图)
    channel_names = {'通道 1 (RD/Range)', '通道 2 (Track Range)', '通道 3 (Track Velocity)'};
    for i = 1:3
        subplot(1, 3, i);
        imagesc(squeeze(data(i, :, :)));
        colorbar;
        title(channel_names{i});
        xlabel('特征单元 (64)'); ylabel('时间帧 (32)');
    end
else
    % 情况 B: 单通道数据 [32 x 64]
    imagesc(data);
    colorbar;
    title(['单通道特征图: ', file]);
    xlabel('特征单元 (64)'); ylabel('时间帧 (32)');
end

colormap('jet'); % 使用 jet 色图，对比度最高，适合看弱信号
axis tight;

% 4. 打印元数据信息 (如果存在)
if exist('metadata', 'var')
    fprintf('\n--- 文件元数据 ---\n');
    disp(metadata);
end