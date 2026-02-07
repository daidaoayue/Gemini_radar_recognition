%% 数据清洗脚本 v4 (兼容旧版MATLAB)
% 功能：检测并删除有问题的航迹、点迹和原始回波数据

clear; clc; close all;

% 关闭警告
warning('off', 'all');

%% 参数设置
MIN_POINTS = 5;  % 最少航迹点数

%% 1. 选择数据目录
fprintf('=== 数据清洗工具 v4 ===\n\n');
fprintf('最少点数要求: %d\n\n', MIN_POINTS);

raw_data_folder = uigetdir(pwd, '选择原始回波文件夹');
if raw_data_folder == 0, error('未选择原始回波文件夹！'); end

track_folder = uigetdir(pwd, '选择航迹文件夹（Tracks_xxx.txt）');
if track_folder == 0, error('未选择航迹文件夹！'); end

point_folder = uigetdir(pwd, '选择点迹文件夹（可取消跳过）');
if point_folder == 0
    point_folder = '';
    fprintf('未选择点迹文件夹\n');
end

%% 2. 扫描文件
fprintf('\n正在扫描文件...\n');

raw_files = dir(fullfile(raw_data_folder, '*_Label_*.dat'));
fprintf('找到 %d 个原始回波文件\n', length(raw_files));

track_files = dir(fullfile(track_folder, 'Tracks_*.txt'));
fprintf('找到 %d 个航迹文件\n', length(track_files));

if ~isempty(point_folder)
    point_files = dir(fullfile(point_folder, 'PointTracks_*.txt'));
    fprintf('找到 %d 个点迹文件\n', length(point_files));
else
    point_files = [];
end

%% 3. 建立索引
fprintf('\n正在建立文件索引...\n');

raw_index = containers.Map();
for i = 1:length(raw_files)
    tokens = regexp(raw_files(i).name, '(\d+)_Label_(\d+)', 'tokens');
    if ~isempty(tokens)
        key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
        raw_index(key) = fullfile(raw_data_folder, raw_files(i).name);
    end
end

track_index = containers.Map();
for i = 1:length(track_files)
    tokens = regexp(track_files(i).name, 'Tracks_(\d+)_(\d+)_', 'tokens');
    if ~isempty(tokens)
        key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
        track_index(key) = fullfile(track_folder, track_files(i).name);
    end
end

point_index = containers.Map();
if ~isempty(point_folder)
    for i = 1:length(point_files)
        tokens = regexp(point_files(i).name, 'PointTracks_(\d+)_(\d+)_', 'tokens');
        if ~isempty(tokens)
            key = sprintf('%d_%d', str2double(tokens{1}{1}), str2double(tokens{1}{2}));
            point_index(key) = fullfile(point_folder, point_files(i).name);
        end
    end
end

fprintf('索引建立完成\n');

%% 4. 检查数据质量
fprintf('\n正在检查数据质量...\n');

problem_tracks = {};
problem_details = {};

all_keys = keys(track_index);
total_tracks = length(all_keys);

h = waitbar(0, '正在检查数据...');

for i = 1:total_tracks
    key = all_keys{i};
    track_path = track_index(key);
    
    problems = {};
    
    % 检查航迹文件
    try
        % 使用简单的方式读取CSV文件
        data = read_csv_simple(track_path);
        
        n_points = size(data, 1);
        
        if n_points < MIN_POINTS
            problems{end+1} = sprintf('航迹点数不足(%d<%d)', n_points, MIN_POINTS);
        else
            % 检查最后两点NaN
            if n_points >= 2
                last_two = data(end-1:end, :);
                if any(isnan(last_two(:)))
                    problems{end+1} = '航迹最后两点存在NaN值';
                end
            end
            
            % 检查时间单调性（第1列是时间，转换为秒）
            time_seconds = data(:, 1);
            if ~isempty(time_seconds)
                time_diff = diff(time_seconds);
                if any(time_diff <= 0)
                    problems{end+1} = '航迹时间非单调递增';
                end
            end
            
            % 检查整体NaN比例
            nan_ratio = sum(isnan(data(:))) / numel(data);
            if nan_ratio > 0.1
                problems{end+1} = sprintf('航迹NaN值过多(%.1f%%)', nan_ratio*100);
            end
        end
    catch ME
        problems{end+1} = sprintf('航迹读取失败: %s', ME.message);
    end
    
    % 检查点迹文件
    if isKey(point_index, key)
        point_path = point_index(key);
        try
            data_p = read_csv_simple(point_path);
            n_points_p = size(data_p, 1);
            
            if n_points_p < MIN_POINTS
                problems{end+1} = sprintf('点迹点数不足(%d<%d)', n_points_p, MIN_POINTS);
            elseif n_points_p >= 2
                last_two_p = data_p(end-1:end, :);
                if any(isnan(last_two_p(:)))
                    problems{end+1} = '点迹最后两点存在NaN值';
                end
            end
        catch ME
            problems{end+1} = sprintf('点迹读取失败: %s', ME.message);
        end
    end
    
    % 检查原始回波
    if ~isKey(raw_index, key)
        problems{end+1} = '无对应原始回波数据';
    end
    
    if ~isempty(problems)
        problem_tracks{end+1} = key;
        problem_details{end+1} = problems;
    end
    
    if mod(i, 50) == 0
        waitbar(i/total_tracks, h, sprintf('检查进度: %d/%d', i, total_tracks));
    end
end

close(h);
warning('on', 'all');

%% 5. 显示结果
fprintf('\n========================================\n');
fprintf('数据检查结果\n');
fprintf('========================================\n');
fprintf('总航迹数: %d\n', total_tracks);
fprintf('问题航迹数: %d\n', length(problem_tracks));
fprintf('正常航迹数: %d\n', total_tracks - length(problem_tracks));
fprintf('问题比例: %.2f%%\n', 100 * length(problem_tracks) / total_tracks);

if ~isempty(problem_tracks)
    fprintf('\n问题航迹详情（前20条）:\n');
    fprintf('----------------------------------------\n');
    
    for i = 1:min(20, length(problem_tracks))
        fprintf('航迹 %s:\n', problem_tracks{i});
        for j = 1:length(problem_details{i})
            fprintf('  - %s\n', problem_details{i}{j});
        end
    end
    
    if length(problem_tracks) > 20
        fprintf('\n... 还有 %d 条问题航迹\n', length(problem_tracks) - 20);
    end
    
    % 统计问题类型
    problem_types = containers.Map();
    for i = 1:length(problem_tracks)
        for j = 1:length(problem_details{i})
            prob = problem_details{i}{j};
            if contains(prob, 'NaN')
                ptype = 'NaN值问题';
            elseif contains(prob, '单调')
                ptype = '时间非单调';
            elseif contains(prob, '回波')
                ptype = '无回波数据';
            elseif contains(prob, '读取失败')
                ptype = '读取失败';
            elseif contains(prob, '点数不足')
                ptype = '点数不足';
            else
                ptype = '其他';
            end
            
            if isKey(problem_types, ptype)
                problem_types(ptype) = problem_types(ptype) + 1;
            else
                problem_types(ptype) = 1;
            end
        end
    end
    
    fprintf('\n问题类型统计:\n');
    fprintf('----------------------------------------\n');
    type_keys = keys(problem_types);
    for i = 1:length(type_keys)
        fprintf('  %s: %d条\n', type_keys{i}, problem_types(type_keys{i}));
    end
end

%% 6. 保存报告
[parent_dir, ~, ~] = fileparts(raw_data_folder);
report_path = fullfile(parent_dir, 'data_quality_report.txt');
fid = fopen(report_path, 'w');

fprintf(fid, '数据质量检查报告\n');
fprintf(fid, '生成时间: %s\n', datestr(now));
fprintf(fid, '最少点数要求: %d\n', MIN_POINTS);
fprintf(fid, '==========================================\n\n');
fprintf(fid, '统计:\n');
fprintf(fid, '  总航迹数: %d\n', total_tracks);
fprintf(fid, '  问题航迹数: %d\n', length(problem_tracks));
fprintf(fid, '  正常航迹数: %d\n', total_tracks - length(problem_tracks));
fprintf(fid, '  问题比例: %.2f%%\n\n', 100 * length(problem_tracks) / total_tracks);

if ~isempty(problem_tracks)
    fprintf(fid, '问题航迹列表:\n');
    fprintf(fid, '==========================================\n');
    for i = 1:length(problem_tracks)
        fprintf(fid, '\n航迹 %s:\n', problem_tracks{i});
        for j = 1:length(problem_details{i})
            fprintf(fid, '  - %s\n', problem_details{i}{j});
        end
    end
end

fclose(fid);
fprintf('\n报告已保存到: %s\n', report_path);

%% 7. 询问删除
if isempty(problem_tracks)
    fprintf('\n✓ 没有发现问题数据！\n');
    return;
end

choice = questdlg(sprintf('发现 %d 条问题航迹，是否删除？', length(problem_tracks)), ...
    '确认', '删除', '取消', '取消');

if strcmp(choice, '删除')
    fprintf('\n正在删除...\n');
    deleted = 0;
    
    for i = 1:length(problem_tracks)
        key = problem_tracks{i};
        
        if isKey(raw_index, key)
            try, delete(raw_index(key)); deleted = deleted + 1; catch, end
        end
        if isKey(track_index, key)
            try, delete(track_index(key)); deleted = deleted + 1; catch, end
        end
        if isKey(point_index, key)
            try, delete(point_index(key)); deleted = deleted + 1; catch, end
        end
    end
    
    fprintf('删除完成！共删除 %d 个文件\n', deleted);
else
    fprintf('已取消\n');
end

fprintf('\n完成！\n');


%% ========== 辅助函数 ==========

function data = read_csv_simple(filepath)
    % 简单读取CSV文件，兼容旧版MATLAB
    % 返回数值矩阵（跳过表头，时间转换为秒）
    
    fid = fopen(filepath, 'r');
    if fid == -1
        error('无法打开文件');
    end
    
    % 跳过表头
    fgetl(fid);
    
    % 读取所有行
    lines = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(strtrim(line))
            lines{end+1} = line;
        end
    end
    fclose(fid);
    
    if isempty(lines)
        data = [];
        return;
    end
    
    % 解析数据
    n_lines = length(lines);
    
    % 确定列数
    first_parts = strsplit(lines{1}, ',');
    n_cols = length(first_parts);
    
    data = zeros(n_lines, n_cols);
    
    for i = 1:n_lines
        parts = strsplit(lines{i}, ',');
        
        % 第1列是时间 (HH:MM:SS.ffffff)
        time_str = strtrim(parts{1});
        time_parts = strsplit(time_str, ':');
        if length(time_parts) == 3
            h = str2double(time_parts{1});
            m = str2double(time_parts{2});
            s = str2double(time_parts{3});
            data(i, 1) = h * 3600 + m * 60 + s;
        else
            data(i, 1) = NaN;
        end
        
        % 其余列是数值
        for j = 2:min(n_cols, length(parts))
            val = str2double(strtrim(parts{j}));
            if isnan(val)
                data(i, j) = NaN;
            else
                data(i, j) = val;
            end
        end
    end
end