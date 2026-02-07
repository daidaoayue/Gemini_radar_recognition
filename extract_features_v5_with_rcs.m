%% 增强版航迹特征提取脚本 V5 (含RCS/多普勒谱宽)
% ========================================================================
% 基于V4的28维特征 + 新增18维RCS/多普勒谱宽特征 = 46维
% ========================================================================
%
% 【V4原有特征 1-28维】
%   1-20: 基础航迹统计特征
%   21-28: 运动稳定性特征
%
% 【V5新增特征 29-46维】RCS和多普勒谱宽
%   29. rcs_mean          - RCS功率均值(dB)
%   30. rcs_peak_mean     - 峰值功率均值(dB)
%   31. rcs_std           - RCS功率标准差 ★鸟类区分关键
%   32. rcs_peak_std      - 峰值功率标准差
%   33. rcs_range         - RCS动态范围 ★鸟类区分关键
%   34. rcs_peak_range    - 峰值动态范围
%   35. rcs_diff_mean     - 帧间RCS变化均值
%   36. rcs_diff_max      - 帧间RCS变化最大值
%   37. rcs_cv            - RCS变异系数 ★鸟类区分关键
%   38. doppler_width_mean    - 多普勒谱宽均值 ★鸟类区分关键
%   39. doppler_width_std     - 多普勒谱宽标准差
%   40. doppler_width_max     - 多普勒谱宽最大值
%   41. bandwidth_3db_mean    - 3dB带宽均值
%   42. bandwidth_3db_std     - 3dB带宽标准差
%   43. energy_conc_mean      - 能量集中度均值 ★鸟类区分关键
%   44. energy_conc_std       - 能量集中度标准差
%   45. peak_mean_ratio       - 峰均比均值
%   46. peak_mean_ratio_std   - 峰均比标准差
%
% 输出: track_data [12, 16] + track_stats [1, 46]

clear; clc; close all;

%% 1. 路径设置
fprintf('=== 航迹特征提取 V5 (含RCS/多普勒) ===\n\n');

rd_root = uigetdir(pwd, '选择RD数据根目录（包含train/val子目录）');
if rd_root == 0; return; end

track_txt_folder = uigetdir(pwd, '选择航迹txt文件夹（Tracks_xxx.txt）');
if track_txt_folder == 0; return; end

raw_data_folder = uigetdir(pwd, '选择原始回波文件夹（*_Label_*.dat）用于提取RCS');
if raw_data_folder == 0
    raw_data_folder = '';
    fprintf('⚠ 未选择原始回波文件夹，RCS特征将为0\n');
end

point_txt_folder = uigetdir(pwd, '选择点迹文件夹（可取消跳过）');
if point_txt_folder == 0
    point_txt_folder = '';
end

output_folder = uigetdir(pwd, '选择输出目录');
if output_folder == 0; return; end

%% 2. 参数设置
TARGET_LEN = 16;
SMOOTH_WIN = 3;
c = 3e8;

%% 3. 预加载数据文件
fprintf('\n正在预加载数据文件...\n');

% 航迹文件
track_files = dir(fullfile(track_txt_folder, 'Tracks_*.txt'));
track_cache = containers.Map();
for i = 1:length(track_files)
    filename = track_files(i).name;
    tokens = regexp(filename, 'Tracks_(\d+)_(\d+)_', 'tokens');
    if ~isempty(tokens)
        key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
        try
            filepath = fullfile(track_txt_folder, filename);
            opts = detectImportOptions(filepath);
            opts.VariableNamingRule = 'preserve';
            track_cache(key) = readtable(filepath, opts);
        catch
        end
    end
end
fprintf('  航迹: %d 条\n', track_cache.Count);

% 点迹文件
point_cache = containers.Map();
if ~isempty(point_txt_folder)
    point_files = dir(fullfile(point_txt_folder, 'PointTracks_*.txt'));
    for i = 1:length(point_files)
        filename = point_files(i).name;
        tokens = regexp(filename, 'PointTracks_(\d+)_(\d+)_', 'tokens');
        if ~isempty(tokens)
            key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
            try
                filepath = fullfile(point_txt_folder, filename);
                opts = detectImportOptions(filepath);
                opts.VariableNamingRule = 'preserve';
                point_cache(key) = readtable(filepath, opts);
            catch
            end
        end
    end
    fprintf('  点迹: %d 条\n', point_cache.Count);
end

% 原始回波文件索引
dat_index = containers.Map();
if ~isempty(raw_data_folder)
    dat_files = dir(fullfile(raw_data_folder, '*_Label_*.dat'));
    for i = 1:length(dat_files)
        tokens = regexp(dat_files(i).name, '(\d+)_Label_(\d+)', 'tokens');
        if ~isempty(tokens)
            key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
            dat_index(key) = fullfile(raw_data_folder, dat_files(i).name);
        end
    end
    fprintf('  原始回波: %d 个\n', dat_index.Count);
end

%% 4. 遍历RD文件并提取特征
splits = {'train', 'val', 'test'};
total_processed = 0;
total_failed = 0;
rcs_extracted = 0;

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
                
                % 获取航迹和点迹数据
                track_table = track_cache(cache_key);
                point_table = [];
                if isKey(point_cache, cache_key)
                    point_table = point_cache(cache_key);
                end
                
                % 获取原始回波路径
                dat_path = '';
                if isKey(dat_index, cache_key)
                    dat_path = dat_index(cache_key);
                end
                
                % 提取特征（28维航迹 + 18维RCS/Doppler = 46维）
                [track_data, track_stats, has_rcs] = extract_features_v5(...
                    track_table, point_table, dat_path, ...
                    point_start, point_end, TARGET_LEN, SMOOTH_WIN);
                
                if isempty(track_data)
                    total_failed = total_failed + 1;
                    continue;
                end
                
                % 保存
                [~, name_only] = fileparts(rd_filename);
                save_path = fullfile(out_label_dir, [name_only, '_track.mat']);
                save(save_path, 'track_data', 'track_stats');
                
                total_processed = total_processed + 1;
                if has_rcs
                    rcs_extracted = rcs_extracted + 1;
                end
                
                if mod(total_processed, 200) == 0
                    fprintf('  已处理 %d 个...\n', total_processed);
                end
                
            catch ME
                fprintf('  [X] %s: %s\n', rd_filename, ME.message);
                total_failed = total_failed + 1;
            end
        end
    end
end

fprintf('\n========================================\n');
fprintf('处理完成!\n');
fprintf('  成功: %d, 失败: %d\n', total_processed, total_failed);
fprintf('  RCS特征提取: %d (%.1f%%)\n', rcs_extracted, 100*rcs_extracted/max(total_processed,1));
fprintf('  输出目录: %s\n', output_folder);
fprintf('  时序特征: [12 x %d]\n', TARGET_LEN);
fprintf('  统计特征: [1 x 46] (28维航迹 + 18维RCS/Doppler)\n');


%% ========================================================================
%  V5特征提取函数（整合RCS/多普勒谱宽）
%  ========================================================================
function [track_data, track_stats, has_rcs] = extract_features_v5(track_T, point_T, dat_path, point_start, point_end, target_len, smooth_win)
    track_data = [];
    track_stats = [];
    has_rcs = false;
    
    n_rows = height(track_T);
    if n_rows < 2
        return;
    end
    
    idx_start = max(1, point_start);
    idx_end = min(n_rows, point_end);
    if idx_start >= idx_end
        idx_start = 1;
        idx_end = n_rows;
    end
    
    segment = track_T(idx_start:idx_end, :);
    n_pts = height(segment);
    if n_pts < 2
        return;
    end
    
    % ============ 提取航迹基础特征 ============
    range_val = get_column(segment, {'滤波距离', 'Range'}, 3);
    azim = get_column(segment, {'滤波方位', 'Azimuth'}, 4);
    pitch = get_column(segment, {'滤波俯仰', 'Pitch'}, 5);
    vel = get_column(segment, {'全速度', 'Vel'}, 6);
    vx = get_column(segment, {'X向速度', 'Vx'}, 7);
    vy = get_column(segment, {'Y向速度', 'Vy'}, 8);
    vz = get_column(segment, {'Z向速度', 'Vz'}, 9);
    heading = get_column(segment, {'航向', 'Heading'}, 10);
    
    dt = 3.0;
    d_range = [0; diff(range_val)] / dt;
    d_azim = [0; diff(azim)] / dt;
    d_pitch = [0; diff(pitch)] / dt;
    d_vel = [0; diff(vel)] / dt;
    
    d_heading = [0; diff(heading)];
    d_heading(abs(d_heading) > 180) = d_heading(abs(d_heading) > 180) - sign(d_heading(abs(d_heading) > 180)) * 360;
    
    % 点迹特征
    amplitude = zeros(n_pts, 1);
    snr = zeros(n_pts, 1);
    point_count = zeros(n_pts, 1);
    
    if ~isempty(point_T) && height(point_T) >= idx_end
        point_seg = point_T(idx_start:min(idx_end, height(point_T)), :);
        if height(point_seg) == n_pts
            amplitude = get_column(point_seg, {'幅度', 'Amplitude'}, 7);
            snr = get_column(point_seg, {'信噪比', 'SNR'}, 8);
            point_count = get_column(point_seg, {'原始点数量', 'PointCount'}, 9);
        end
    end
    
    amplitude(amplitude > 0) = log10(amplitude(amplitude > 0) + 1);
    snr(snr > 0) = log10(snr(snr > 0) + 1);
    
    curvature = abs(d_heading) ./ (vel + 0.1);
    
    % ============ 时序特征 [12 x target_len] ============
    features = [
        vel(:)'; vz(:)'; d_vel(:)'; d_range(:)';
        d_azim(:)'; d_pitch(:)'; heading(:)'; range_val(:)';
        pitch(:)'; amplitude(:)'; snr(:)'; curvature(:)';
    ];
    
    if n_pts >= smooth_win
        for ch = 1:12
            features(ch, :) = movmedian(features(ch, :), smooth_win, 'omitnan');
        end
    end
    
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
    
    for ch = 1:12
        row = track_data(ch, :);
        if std(row) > 1e-6
            track_data(ch, :) = (row - mean(row)) / std(row);
        end
    end
    track_data(isnan(track_data)) = 0;
    
    % ============ 统计特征 [1 x 46] ============
    track_stats = zeros(1, 46);
    
    % --- 原有28维特征 (V4) ---
    track_stats(1) = mean(vel);
    track_stats(2) = std(vel);
    track_stats(3) = max(vel);
    track_stats(4) = min(vel);
    track_stats(5) = mean(vz);
    track_stats(6) = std(vz);
    track_stats(7) = mean(abs(d_vel));
    track_stats(8) = max(abs(d_vel));
    track_stats(9) = mean(abs(d_heading));
    track_stats(10) = std(d_heading);
    track_stats(11) = mean(range_val);
    track_stats(12) = range_val(end) - range_val(1);
    track_stats(13) = mean(pitch);
    track_stats(14) = std(pitch);
    track_stats(15) = mean(amplitude);
    track_stats(16) = std(amplitude);
    track_stats(17) = mean(snr);
    track_stats(18) = mean(point_count);
    track_stats(19) = n_pts;
    track_stats(20) = sum(sqrt(diff(range_val).^2));
    
    % 21-28: 运动稳定性特征
    stability_score = (std(vel)/(mean(vel)+0.1)) + (mean(abs(d_heading))/10) + (mean(abs(d_vel))/(mean(vel)+0.1));
    track_stats(21) = stability_score;
    track_stats(22) = mean(curvature);
    track_stats(23) = max(curvature);
    track_stats(24) = std(curvature);
    track_stats(25) = std(vel) / (mean(vel) + 0.1);
    track_stats(26) = mean(abs(vz) ./ (vel + 0.1));
    
    if n_pts >= 4
        vel_fft = abs(fft(vel - mean(vel)));
        half_len = floor(n_pts / 2);
        if half_len > 1
            track_stats(27) = max(vel_fft(2:half_len));
        end
    end
    
    if n_pts > 2
        d_range_signs = sign(d_range(2:end));
        track_stats(28) = abs(sum(d_range_signs)) / length(d_range_signs);
    else
        track_stats(28) = 0.5;
    end
    
    % --- 新增18维RCS/多普勒特征 (29-46) ---
    rcs_features = zeros(1, 18);
    
    if ~isempty(dat_path) && exist(dat_path, 'file')
        try
            rcs_features = extract_rcs_doppler_from_dat(dat_path, point_start, point_end);
            has_rcs = true;
        catch
            % 提取失败，保持为0
        end
    end
    
    track_stats(29:46) = rcs_features;
    track_stats(isnan(track_stats)) = 0;
end


%% ========================================================================
%  从原始回波提取RCS和多普勒谱宽特征
%  ========================================================================
function features = extract_rcs_doppler_from_dat(dat_path, point_start, point_end)
    features = zeros(1, 18);
    
    % 解析.dat文件
    frames = parse_dat_file(dat_path);
    
    if isempty(frames)
        return;
    end
    
    % 筛选指定航迹点范围的帧
    selected = {};
    for i = 1:length(frames)
        pt = frames{i}.track_point;
        if pt >= point_start && pt <= point_end
            selected{end+1} = frames{i};
        end
    end
    
    if length(selected) < 2
        selected = frames;  % 如果筛选后太少，使用全部
    end
    
    if isempty(selected)
        return;
    end
    
    n_frames = length(selected);
    
    % ============ RCS特征 (1-9) ============
    frame_powers = zeros(n_frames, 1);
    peak_powers = zeros(n_frames, 1);
    
    for i = 1:n_frames
        iq = selected{i}.iq_data;
        power = abs(iq).^2;
        frame_powers(i) = sum(power(:));
        peak_powers(i) = max(power(:));
    end
    
    frame_powers_db = 10 * log10(frame_powers + 1e-10);
    peak_powers_db = 10 * log10(peak_powers + 1e-10);
    
    features(1) = mean(frame_powers_db);      % rcs_mean
    features(2) = mean(peak_powers_db);       % rcs_peak_mean
    features(3) = std(frame_powers_db);       % rcs_std ★
    features(4) = std(peak_powers_db);        % rcs_peak_std
    features(5) = max(frame_powers_db) - min(frame_powers_db);  % rcs_range ★
    features(6) = max(peak_powers_db) - min(peak_powers_db);    % rcs_peak_range
    
    if n_frames > 1
        features(7) = mean(abs(diff(frame_powers_db)));  % rcs_diff_mean
        features(8) = max(abs(diff(frame_powers_db)));   % rcs_diff_max
    end
    
    features(9) = std(frame_powers) / (mean(frame_powers) + 1e-10);  % rcs_cv ★
    
    % ============ 多普勒谱宽特征 (10-18) ============
    c = 3e8;
    spectral_widths = [];
    bandwidth_3db = [];
    energy_concs = [];
    peak_ratios = [];
    
    for i = 1:n_frames
        iq = selected{i}.iq_data;
        prt = selected{i}.prt;
        freq = selected{i}.freq;
        prt_num = selected{i}.prt_num;
        
        % 计算RD矩阵
        delta_Vr = c / (2 * prt_num * prt * freq);
        velocity_num = 2^(ceil(log2(prt_num * delta_Vr / 0.1)));
        vr = (-velocity_num/2 : velocity_num/2-1) * delta_Vr * prt_num / velocity_num;
        
        mtd_win = taylorwin(prt_num, 4, -30);
        data_win = iq .* mtd_win';
        data_win = data_win - mean(data_win, 2);
        rd = abs(fftshift(fft(data_win, velocity_num, 2), 2));
        
        % 对每个距离门计算特征
        for r_idx = 1:size(rd, 1)
            spectrum = rd(r_idx, :);
            spec_norm = spectrum / (sum(spectrum) + 1e-10);
            
            % 谱标准差
            mean_v = sum(vr .* spec_norm);
            var_v = sum((vr - mean_v).^2 .* spec_norm);
            spectral_widths(end+1) = sqrt(var_v);
            
            % 3dB带宽
            peak_val = max(spec_norm);
            above = spec_norm > peak_val/2;
            if sum(above) > 0
                vr_above = vr(above);
                bandwidth_3db(end+1) = max(vr_above) - min(vr_above);
            else
                bandwidth_3db(end+1) = 0;
            end
            
            % 能量集中度
            [~, peak_idx] = max(spectrum);
            win = 5;
            s_idx = max(1, peak_idx - win);
            e_idx = min(length(spectrum), peak_idx + win);
            energy_concs(end+1) = sum(spectrum(s_idx:e_idx)) / (sum(spectrum) + 1e-10);
            
            % 峰均比
            peak_ratios(end+1) = peak_val / (mean(spec_norm) + 1e-10);
        end
    end
    
    features(10) = mean(spectral_widths);     % doppler_width_mean ★
    features(11) = std(spectral_widths);      % doppler_width_std
    features(12) = max(spectral_widths);      % doppler_width_max
    features(13) = mean(bandwidth_3db);       % bandwidth_3db_mean
    features(14) = std(bandwidth_3db);        % bandwidth_3db_std
    features(15) = mean(energy_concs);        % energy_conc_mean ★
    features(16) = std(energy_concs);         % energy_conc_std
    features(17) = mean(peak_ratios);         % peak_mean_ratio
    features(18) = std(peak_ratios);          % peak_mean_ratio_std
end


%% ========================================================================
%  解析原始回波.dat文件
%  ========================================================================
function frames = parse_dat_file(filepath)
    frames = {};
    
    fid = fopen(filepath, 'r');
    if fid == -1
        return;
    end
    
    frame_head = hex2dec('FA55FA55');
    frame_end = hex2dec('55FA55FA');
    
    while ~feof(fid)
        % 查找帧头
        head_find = fread(fid, 1, 'uint32');
        if isempty(head_find)
            break;
        end
        
        while head_find ~= frame_head && ~feof(fid)
            fseek(fid, -3, 'cof');
            head_find = fread(fid, 1, 'uint32');
        end
        
        if feof(fid)
            break;
        end
        
        % 读取帧长度
        frame_len = fread(fid, 1, 'uint32');
        if isempty(frame_len)
            break;
        end
        
        try
            % 读取头部参数（11个uint32）
            header = fread(fid, 11, 'uint32');
            if length(header) < 11
                break;
            end
            
            track_point = header(5);
            freq = header(7) * 1e6;
            prt_num = header(9);
            prt = header(10) * 0.0125e-6;
            
            % 读取IQ数据
            n_iq = prt_num * 31 * 2;
            iq_data = fread(fid, n_iq, 'float');
            
            if length(iq_data) < n_iq
                break;
            end
            
            % 组织复数数据 [31 x prt_num]
            iq_real = iq_data(1:2:end);
            iq_imag = iq_data(2:2:end);
            iq_matrix = reshape(iq_real + 1i * iq_imag, 31, prt_num);
            
            frame_info = struct();
            frame_info.track_point = track_point;
            frame_info.freq = freq;
            frame_info.prt = prt;
            frame_info.prt_num = prt_num;
            frame_info.iq_data = iq_matrix;
            
            frames{end+1} = frame_info;
            
            % 跳过帧尾
            fseek(fid, 4, 'cof');
            
        catch
            fseek(fid, frame_len * 4 - 52, 'cof');
        end
    end
    
    fclose(fid);
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