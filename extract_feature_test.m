%% 航迹特征提取 PRO 版 (修复插值报错)：强壮的 NaN 处理
% 功能：
% 1. 递归扫描数据集，精准切片。
% 2. 【修复】针对包含 NaN 的数据片段，自动过滤无效点再插值。
% 3. 结果统一保存在指定目录。

clear; clc; close all;

%% 1. 路径配置
% 1.1 选择 RD 数据集根目录
rd_root_dir = uigetdir(pwd, '1/3 选择 RD 数据集根目录 (包含 train/val 的上一级)');
if rd_root_dir == 0, return; end

% 1.2 选择原始航迹 txt 文件夹
track_txt_dir = uigetdir(pwd, '2/3 选择原始航迹 TXT 文件夹');
if track_txt_dir == 0, return; end

% 1.3 选择结果保存目录
output_save_dir = uigetdir(pwd, '3/3 选择特征矩阵保存目录 (用于验证)');
if output_save_dir == 0, return; end

fprintf('📂 结果将保存到: %s\n', output_save_dir);

%% 2. 预加载所有航迹 (构建内存数据库)
fprintf('🚀 正在预加载所有航迹文件到内存...\n');
track_map = containers.Map('KeyType','int32','ValueType','any');
txt_files = dir(fullfile(track_txt_dir, 'Tracks_*.txt'));

hWait = waitbar(0, '正在加载航迹数据库...');
for i = 1:length(txt_files)
    fname = txt_files(i).name;
    tokens = regexp(fname, 'Tracks_(\d+)', 'tokens');
    if isempty(tokens), continue; end
    track_id = int32(str2double(tokens{1}{1}));
    
    fpath = fullfile(track_txt_dir, fname);
    try
        opts = detectImportOptions(fpath);
        opts.VariableNamingRule = 'preserve';
        T = readtable(fpath, opts); 
        track_map(track_id) = T;
    catch
        fprintf('⚠️ 无法读取 %s\n', fname);
    end
    if mod(i, 50) == 0, waitbar(i/length(txt_files), hWait); end
end
close(hWait);
fprintf('✅ 成功加载 %d 条航迹数据。\n', track_map.Count);

%% 3. 递归遍历 RD 数据集并处理
fprintf('🔥 开始递归搜索并处理...\n');

all_mat_files = dir(fullfile(rd_root_dir, '**/*.mat'));

count_success = 0;
count_fail = 0;
count_nan = 0; % 统计因 NaN 过多跳过的文件

hWait = waitbar(0, '正在处理数据切片...');
total_files = length(all_mat_files);

for i = 1:total_files
    rd_file = all_mat_files(i);
    rd_name = rd_file.name;
    
    if contains(rd_name, '_track.mat') || contains(rd_name, '_motion.mat')
        continue;
    end
    
    % --- Step 1: 解析文件名 ---
    pat = 'Track(\d+)_.*Points(\d+)-(\d+)';
    tokens = regexp(rd_name, pat, 'tokens');
    
    if isempty(tokens), continue; end
    
    tid = int32(str2double(tokens{1}{1}));
    p_start = str2double(tokens{1}{2});
    p_end = str2double(tokens{1}{3});
    
    if ~isKey(track_map, tid)
        count_fail = count_fail + 1;
        continue;
    end
    
    T = track_map(tid);
    
% --- Step 2: 截取数据片段 (上下文增强版) ---
    % 核心策略：以 RD 的结束时间为准，往前追溯一段历史（例如 64 个点）
    % 这样既保证了包含当前的动作，又提供了足够的历史来计算机动性
    HISTORY_LEN = 64; % 回溯 64 个点 (约 3-6 秒)，足够看清动作模式
    
    idx_e = min(height(T), p_end);
    idx_s = max(1, idx_e - HISTORY_LEN); % 从结束点往前推
    
    % 如果即使往前推，总长度还是很短（比如刚起飞），那就只能认了
    % 但大多数情况下，这会提供丰富的历史特征
    
    if idx_s > idx_e
        count_fail = count_fail + 1;
        continue; 
    end
    
    % 提取原始列
    r = pickCol(T, {'滤波距离','Range'}, 3);
    v = pickCol(T, {'全速度','Vel'}, 6);
    vz = pickCol(T, {'Z向速度','Vz'}, 9);
    heading = pickCol(T, {'航向','Heading'}, 10);
    
    % 切片
    seg_v = v(idx_s:idx_e);
    seg_vz = vz(idx_s:idx_e);
    seg_h = heading(idx_s:idx_e);
    
    TARGET_LEN = 128;
    
    % --- Step 3: 特征工程 (修复 NaN 报错) ---
    % 定义通用插值函数（内含 NaN 过滤）
    % 原始时间轴 0~1
    raw_len = length(seg_v);
    t_raw = linspace(0, 1, raw_len)'; 
    t_target = linspace(0, 1, TARGET_LEN)';
    
    % 尝试插值三个核心变量
    [i_v, ok1] = safe_interp(t_raw, seg_v, t_target);
    [i_vz, ok2] = safe_interp(t_raw, seg_vz, t_target);
    
    % 航向角特殊处理 (先解卷绕)
    if length(seg_h) >= 2 && sum(~isnan(seg_h)) >= 2
        valid_h = ~isnan(seg_h);
        t_valid = t_raw(valid_h);
        h_valid = seg_h(valid_h);
        
        rad_h = deg2rad(h_valid);
        u = unwrap(rad_h); % 对有效点解卷绕
        
        % 插值解卷绕后的弧度
        if length(u) >= 2
             i_u = interp1(t_valid, u, t_target, 'linear', 'extrap'); % 航向用线性即可，pchip可能过冲
             i_h = rad2deg(i_u);
             ok3 = true;
        else
             i_h = zeros(TARGET_LEN, 1);
             ok3 = false;
        end
    else
        i_h = zeros(TARGET_LEN, 1);
        ok3 = false;
    end
    
    % 如果任意一个关键特征全挂了，就生成全零矩阵
    if ~ok1 && ~ok2
        track_data = zeros(6, TARGET_LEN);
        count_nan = count_nan + 1;
    else
        % 补救措施：如果某个通道挂了但其他没挂，挂的通道补0
        if ~ok1, i_v = zeros(TARGET_LEN,1); end
        if ~ok2, i_vz = zeros(TARGET_LEN,1); end
        
        dt = 1; 
        
        feat_vel = i_v;
        feat_vz = i_vz;
        feat_acc_rad = gradient(i_v, dt); 
        feat_acc_z = gradient(i_vz, dt);
        feat_turn_rate = gradient(i_h, dt);
        feat_jerk = gradient(feat_acc_rad, dt);
        
        % --- Step 4: 物理量级缩放 ---
        s_vel = 30.0;
        s_acc = 5.0;
        s_turn = 5.0;
        s_jerk = 1.0;
        
        track_data = [ ...
            feat_vel' / s_vel; ...
            feat_vz' / s_vel; ...
            feat_acc_rad' / s_acc; ...
            feat_acc_z' / s_acc; ...
            feat_turn_rate' / s_turn; ...
            feat_jerk' / s_jerk ...
        ];
    end
    
    % --- Step 5: 保存结果 ---
    [~, name_core] = fileparts(rd_name);
    save_path = fullfile(output_save_dir, [name_core, '_track.mat']);
    save(save_path, 'track_data');
    
    count_success = count_success + 1;
    
    if mod(i, 200) == 0
        waitbar(i/total_files, hWait, sprintf('处理中... %d/%d (NaN跳过: %d)', i, total_files, count_nan));
    end
end
close(hWait);

fprintf('\n🎉 处理完成！\n');
fprintf('  成功生成: %d\n', count_success);
fprintf('  因数据全是NaN补零: %d\n', count_nan);
fprintf('  结果目录: %s\n', output_save_dir);

%% 辅助函数1: 安全插值 (自动剔除 NaN)
function [out, is_ok] = safe_interp(x, y, xq)
    % 确保是列向量
    x = x(:); y = y(:);
    
    % 找有效点
    valid = ~isnan(y);
    
    if sum(valid) < 2
        out = zeros(length(xq), 1);
        is_ok = false;
    else
        % 仅使用有效点进行插值
        % 使用 'linear' + 'extrap' 避免边界 NaN，或者用 'nearest'
        % 推荐 'pchip' 保持波形，但如果点太少退化为 'linear'
        try
            if sum(valid) >= 4
                out = interp1(x(valid), y(valid), xq, 'pchip', 'extrap');
            else
                out = interp1(x(valid), y(valid), xq, 'linear', 'extrap');
            end
            is_ok = true;
        catch
            out = zeros(length(xq), 1);
            is_ok = false;
        end
    end
end

%% 辅助函数2: 选列
function col = pickCol(T, nameList, idx)
    colNames = T.Properties.VariableNames;
    hit = find(ismember(lower(colNames), lower(nameList)), 1);
    if ~isempty(hit)
        col = T{:, hit};
    elseif idx <= width(T)
        col = T{:, idx};
    else
        col = nan(height(T), 1); % 找不到列时返回 NaN 而不是 0
    end
end