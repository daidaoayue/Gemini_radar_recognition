%% 增强版航迹特征提取脚本 V4
% ========================================================================
% 基于问题航迹分析结果的针对性改进
% ========================================================================
% 
% 【分析发现的问题】
% 低置信度样本的特点：
%   - turn_rate高 (+156.7%) - 转弯多
%   - velocity_std高 (+77.2%) - 速度不稳定  
%   - mean_accel高 (+75.7%) - 加速度大
%   - heading_stability高 (+155.4%) - 航向不稳定
%
% 【改进策略】
% 1. 添加运动稳定性指标 - 让模型知道这是"不稳定"的样本
% 2. 添加频域特征 - 捕捉周期性运动模式（鸟类翅膀拍动）
% 3. 添加运动曲率特征 - 区分盘旋和直线飞行
% 4. 添加运动模式分类特征 - 直接判断运动类型
%
% 输出: track_data [12, 16] + track_stats [1, 28] (原20维+新8维)

clear; clc; close all;

%% 1. 路径设置
rd_root = uigetdir(pwd, '选择RD数据根目录（包含train/val/test子目录）');
if rd_root == 0; return; end

track_txt_folder = uigetdir(pwd, '选择航迹txt文件夹（Tracks_xxx.txt）');
if track_txt_folder == 0; return; end

point_txt_folder = uigetdir(pwd, '选择点迹txt文件夹（可取消跳过）');
if point_txt_folder == 0
    point_txt_folder = '';
    fprintf('未选择点迹文件夹，将只使用航迹数据\n');
end

output_folder = uigetdir(pwd, '选择输出目录');
if output_folder == 0; return; end

%% 2. 参数设置
TARGET_LEN = 16;  % 时序长度
SMOOTH_WIN = 3;   % 平滑窗口

%% 3. 预加载所有航迹和点迹文件
fprintf('正在预加载数据文件...\n');

% 航迹文件
track_files = dir(fullfile(track_txt_folder, 'Tracks_*.txt'));
track_cache = containers.Map();
for i = 1:length(track_files)
    filename = track_files(i).name;
    tokens = regexp(filename, 'Tracks_(\d+)_(\d+)_', 'tokens');
    if ~isempty(tokens)
        track_id = str2double(tokens{1}{1});
        track_label = str2double(tokens{1}{2});
        key = sprintf('%d_%d', track_id, track_label);
        try
            filepath = fullfile(track_txt_folder, filename);
            opts = detectImportOptions(filepath);
            opts.VariableNamingRule = 'preserve';
            T = readtable(filepath, opts);
            track_cache(key) = T;
        catch
        end
    end
end
fprintf('已加载 %d 条航迹数据\n', track_cache.Count);

% 点迹文件
point_cache = containers.Map();
if ~isempty(point_txt_folder)
    point_files = dir(fullfile(point_txt_folder, 'PointTracks_*.txt'));
    for i = 1:length(point_files)
        filename = point_files(i).name;
        tokens = regexp(filename, 'PointTracks_(\d+)_(\d+)_', 'tokens');
        if ~isempty(tokens)
            track_id = str2double(tokens{1}{1});
            track_label = str2double(tokens{1}{2});
            key = sprintf('%d_%d', track_id, track_label);
            try
                filepath = fullfile(point_txt_folder, filename);
                opts = detectImportOptions(filepath);
                opts.VariableNamingRule = 'preserve';
                T = readtable(filepath, opts);
                point_cache(key) = T;
            catch
            end
        end
    end
    fprintf('已加载 %d 条点迹数据\n', point_cache.Count);
end

%% 4. 遍历RD文件并提取特征
splits = {'train', 'val', 'test'};
total_processed = 0;
total_failed = 0;

for s = 1:length(splits)
    split = splits{s};
    split_dir = fullfile(rd_root, split);
    
    if ~exist(split_dir, 'dir')
        continue;
    end
    
    for label = 0:5
        label_dir = fullfile(split_dir, num2str(label));
        if ~exist(label_dir, 'dir')
            continue;
        end
        
        % 创建输出目录
        out_label_dir = fullfile(output_folder, split, num2str(label));
        if ~exist(out_label_dir, 'dir')
            mkdir(out_label_dir);
        end
        
        rd_files = dir(fullfile(label_dir, 'Track*.mat'));
        fprintf('\n处理 %s/%d: %d 个文件\n', split, label, length(rd_files));
        
        for f = 1:length(rd_files)
            rd_filename = rd_files(f).name;
            
            try
                % 解析文件名
                tokens = regexp(rd_filename, 'Track(\d+)_Label(\d+)_Group\d+_Points(\d+)-(\d+)', 'tokens');
                
                if isempty(tokens)
                    total_failed = total_failed + 1;
                    continue;
                end
                
                track_id = str2double(tokens{1}{1});
                file_label = str2double(tokens{1}{2});
                point_start = str2double(tokens{1}{3});
                point_end = str2double(tokens{1}{4});
                
                % 构建缓存键
                cache_key = sprintf('%d_%d', track_id, file_label + 1);
                
                if ~isKey(track_cache, cache_key)
                    cache_key = sprintf('%d_%d', track_id, file_label);
                    if ~isKey(track_cache, cache_key)
                        total_failed = total_failed + 1;
                        continue;
                    end
                end
                
                % 获取数据
                track_table = track_cache(cache_key);
                point_table = [];
                if isKey(point_cache, cache_key)
                    point_table = point_cache(cache_key);
                end
                
                % 提取增强特征
                [track_data, track_stats] = extract_enhanced_features_v4(...
                    track_table, point_table, point_start, point_end, TARGET_LEN, SMOOTH_WIN);
                
                if isempty(track_data)
                    total_failed = total_failed + 1;
                    continue;
                end
                
                % 保存
                [~, name_only] = fileparts(rd_filename);
                save_path = fullfile(out_label_dir, [name_only, '.mat']);
                save(save_path, 'track_data', 'track_stats');
                
                total_processed = total_processed + 1;
                
            catch ME
                fprintf('  [X] %s: %s\n', rd_filename, ME.message);
                total_failed = total_failed + 1;
            end
        end
    end
end

fprintf('\n========================================\n');
fprintf('处理完成!\n');
fprintf('   成功: %d, 失败: %d\n', total_processed, total_failed);
fprintf('   输出目录: %s\n', output_folder);
fprintf('   时序特征维度: [12 x %d]\n', TARGET_LEN);
fprintf('   统计特征维度: [1 x 28] (原20维 + 新8维)\n');


%% ========================================================================
%  增强特征提取函数 V4
%  ========================================================================
function [track_data, track_stats] = extract_enhanced_features_v4(track_T, point_T, point_start, point_end, target_len, smooth_win)
    track_data = [];
    track_stats = [];
    
    n_rows = height(track_T);
    if n_rows < 2
        return;
    end
    
    % 限制范围
    idx_start = max(1, point_start);
    idx_end = min(n_rows, point_end);
    
    if idx_start >= idx_end
        idx_start = 1;
        idx_end = n_rows;
    end
    
    % 提取片段
    segment = track_T(idx_start:idx_end, :);
    n_pts = height(segment);
    
    if n_pts < 2
        return;
    end
    
    % ============ 从航迹提取基础特征 ============
    range_val = get_column(segment, {'滤波距离', 'Range'}, 3);
    azim = get_column(segment, {'滤波方位', 'Azimuth'}, 4);
    pitch = get_column(segment, {'滤波俯仰', 'Pitch'}, 5);
    vel = get_column(segment, {'全速度', 'Vel'}, 6);
    vx = get_column(segment, {'X向速度', 'Vx'}, 7);
    vy = get_column(segment, {'Y向速度', 'Vy'}, 8);
    vz = get_column(segment, {'Z向速度', 'Vz'}, 9);
    heading = get_column(segment, {'航向', 'Heading'}, 10);
    
    % 计算导数特征（变化率）
    dt = 3.0;  % 假设采样间隔约3秒
    d_range = [0; diff(range_val)] / dt;
    d_azim = [0; diff(azim)] / dt;
    d_pitch = [0; diff(pitch)] / dt;
    d_vel = [0; diff(vel)] / dt;  % 加速度
    
    % 航向变化（处理360度跳变）
    d_heading = [0; diff(heading)];
    d_heading(abs(d_heading) > 180) = d_heading(abs(d_heading) > 180) - sign(d_heading(abs(d_heading) > 180)) * 360;
    
    % ============ 从点迹提取特征（如果有）============
    amplitude = zeros(n_pts, 1);
    snr = zeros(n_pts, 1);
    point_count = zeros(n_pts, 1);
    doppler = zeros(n_pts, 1);
    
    if ~isempty(point_T) && height(point_T) >= idx_end
        point_seg = point_T(idx_start:min(idx_end, height(point_T)), :);
        if height(point_seg) == n_pts
            amplitude = get_column(point_seg, {'幅度', 'Amplitude'}, 7);
            snr = get_column(point_seg, {'信噪比', 'SNR'}, 8);
            point_count = get_column(point_seg, {'原始点数量', 'PointCount'}, 9);
            doppler = get_column(point_seg, {'多普勒速度', 'Doppler'}, 6);
        end
    end
    
    % 对幅度和信噪比做对数变换
    amplitude(amplitude > 0) = log10(amplitude(amplitude > 0) + 1);
    snr(snr > 0) = log10(snr(snr > 0) + 1);
    
    % ============ 【新增】运动曲率特征 ============
    % 曲率 = 角速度 / 线速度（区分盘旋和直线飞行）
    curvature = abs(d_heading) ./ (vel + 0.1);
    
    % ============ 组合12个时序特征 ============
    features = [
        vel(:)';           % 1. 全速度
        vz(:)';            % 2. 垂直速度
        d_vel(:)';         % 3. 加速度
        d_range(:)';       % 4. 距离变化率
        d_azim(:)';        % 5. 方位变化率
        d_pitch(:)';       % 6. 俯仰变化率
        heading(:)';       % 7. 航向
        range_val(:)';     % 8. 距离
        pitch(:)';         % 9. 俯仰角
        amplitude(:)';     % 10. 幅度（点迹）
        snr(:)';           % 11. 信噪比（点迹）
        curvature(:)';     % 12. 【改】运动曲率（替换原point_count）
    ];  % [12 x n_pts]
    
    % 平滑
    if n_pts >= smooth_win
        for ch = 1:size(features, 1)
            features(ch, :) = movmedian(features(ch, :), smooth_win, 'omitnan');
        end
    end
    
    % 插值到目标长度
    if n_pts == target_len
        track_data = features;
    elseif n_pts == 1
        track_data = repmat(features, 1, target_len);
    else
        x_old = linspace(0, 1, n_pts);
        x_new = linspace(0, 1, target_len);
        track_data = zeros(12, target_len);
        for ch = 1:12
            track_data(ch, :) = interp1(x_old, features(ch, :), x_new, 'pchip');
        end
    end
    
    % 标准化每个通道
    for ch = 1:12
        row = track_data(ch, :);
        if std(row) > 1e-6
            track_data(ch, :) = (row - mean(row)) / std(row);
        end
    end
    
    % 处理NaN
    track_data(isnan(track_data)) = 0;
    
    % ============ 统计特征（28维 = 原20维 + 新8维）============
    track_stats = zeros(1, 28);
    
    % === 原有20维特征 ===
    % 速度统计
    track_stats(1) = mean(vel);        % 平均速度
    track_stats(2) = std(vel);         % 速度标准差
    track_stats(3) = max(vel);         % 最大速度
    track_stats(4) = min(vel);         % 最小速度
    
    % 垂直速度统计
    track_stats(5) = mean(vz);         % 平均垂直速度
    track_stats(6) = std(vz);          % 垂直速度波动
    
    % 加速度统计
    track_stats(7) = mean(abs(d_vel)); % 平均加速度幅值
    track_stats(8) = max(abs(d_vel));  % 最大加速度
    
    % 航向变化统计
    track_stats(9) = mean(abs(d_heading));  % 平均转弯率
    track_stats(10) = std(d_heading);       % 航向稳定性
    
    % 距离和俯仰统计
    track_stats(11) = mean(range_val);      % 平均距离
    track_stats(12) = range_val(end) - range_val(1);  % 距离变化总量
    track_stats(13) = mean(pitch);          % 平均俯仰
    track_stats(14) = std(pitch);           % 俯仰波动
    
    % 点迹特征统计
    track_stats(15) = mean(amplitude);      % 平均幅度
    track_stats(16) = std(amplitude);       % 幅度波动
    track_stats(17) = mean(snr);            % 平均信噪比
    track_stats(18) = mean(point_count);    % 平均点数
    
    % 轨迹长度和持续时间
    track_stats(19) = n_pts;                % 点数
    track_stats(20) = sum(sqrt(diff(range_val).^2));  % 轨迹长度
    
    % === 【新增】8维特征（针对低置信度样本问题）===
    
    % 21. 运动稳定性指数（综合指标）
    % 低置信度样本的velocity_std, turn_rate, accel都高
    % 这个指数越高，说明运动越不稳定
    stability_score = (std(vel) / (mean(vel) + 0.1)) + ...
                      (mean(abs(d_heading)) / 10) + ...
                      (mean(abs(d_vel)) / (mean(vel) + 0.1));
    track_stats(21) = stability_score;
    
    % 22. 运动曲率均值（区分盘旋和直线）
    track_stats(22) = mean(curvature);
    
    % 23. 运动曲率最大值
    track_stats(23) = max(curvature);
    
    % 24. 运动曲率标准差
    track_stats(24) = std(curvature);
    
    % 25. 速度变异系数（CV = std/mean，归一化的不稳定性）
    track_stats(25) = std(vel) / (mean(vel) + 0.1);
    
    % 26. 垂直运动比例（vz/vel的平均值）
    vz_ratio = abs(vz) ./ (vel + 0.1);
    track_stats(26) = mean(vz_ratio);
    
    % 27. 速度FFT主频幅度（捕捉周期性运动）
    if n_pts >= 4
        vel_centered = vel - mean(vel);
        vel_fft = abs(fft(vel_centered));
        % 取前半部分（排除直流分量）
        half_len = floor(n_pts / 2);
        if half_len > 1
            track_stats(27) = max(vel_fft(2:half_len));
        else
            track_stats(27) = 0;
        end
    else
        track_stats(27) = 0;
    end
    
    % 28. 运动方向一致性（距离是否持续增加或减少）
    % 计算d_range的符号一致性
    if n_pts > 2
        d_range_signs = sign(d_range(2:end));  % 排除第一个0
        consistency = abs(sum(d_range_signs)) / length(d_range_signs);
        track_stats(28) = consistency;  % 1=完全一致（接近或远离），0=来回震荡
    else
        track_stats(28) = 0.5;
    end
    
    % 标准化新增特征（可选，但建议）
    % 这些特征在训练时会被模型自动学习权重
    
    % 处理NaN
    track_stats(isnan(track_stats)) = 0;
end


%% 安全获取列
function col = get_column(T, name_list, fallback_idx)
    col_names = T.Properties.VariableNames;
    
    for i = 1:length(name_list)
        idx = find(strcmpi(col_names, name_list{i}), 1);
        if ~isempty(idx)
            col = T{:, idx};
            col = col(:);
            return;
        end
    end
    
    if fallback_idx <= width(T)
        col = T{:, fallback_idx};
        col = col(:);
    else
        col = zeros(height(T), 1);
    end
end