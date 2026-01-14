%% test_data_processor.m

clear; clc; close all;

%% 1. 设置路径
input_folder = uigetdir(pwd, '选择包含原始数据的根目录');
if input_folder == 0
    error('未选择输入文件夹！');
end

% 验证目录结构
raw_data_folder = fullfile(input_folder, '原始回波');
track_folder = fullfile(input_folder, '航迹');
point_folder = fullfile(input_folder, '点迹');

if ~all([exist(raw_data_folder, 'dir'), exist(track_folder, 'dir'), exist(point_folder, 'dir')])
    error('错误！根目录下需包含原始回波、点迹、航迹文件夹。');
end

output_base = uigetdir(pwd, '选择输出文件夹');
if output_base == 0
    error('未选择输出文件夹！');
end


% 创建测试集输出目录
test_output_dir = fullfile(output_base, 'test');
if ~exist(test_output_dir, 'dir')
    mkdir(test_output_dir);
end

%% 2. 设置参数
Fs = 20e6;
c = 3e8;
delta_R = c/2/Fs;

%% 3. 获取所有测试集.dat
%文件名字为'*.dat'
dat_files = dir(fullfile(raw_data_folder, '*.dat'));

test_dat_files = [];
for i = 1:length(dat_files)
    if ~contains(dat_files(i).name, 'Label')
        test_dat_files = [test_dat_files; dat_files(i)];
    end
end

if isempty(test_dat_files)
    % 如果没有找到不含Label的文件，就使用所有.dat文件
    test_dat_files = dat_files;
end

fprintf('找到 %d 个测试数据文件\n', length(test_dat_files));

%% 4. 初始化统计变量
total_processed = 0;
total_matrices = 0;
failed_files = {};
track_id_list = [];  % 记录所有处理的航迹ID

%% 5. 处理每个测试文件
for file_idx = 1:length(test_dat_files)
    try
        dat_filename = test_dat_files(file_idx).name;
        fprintf('\n处理文件 [%d/%d]: %s\n', file_idx, length(test_dat_files), dat_filename);
        
        [~, name_without_ext, ~] = fileparts(dat_filename);

        track_id_match = regexp(name_without_ext, '^(\d+)', 'tokens');
        if isempty(track_id_match)
            warning('无法从文件名 %s 提取track_id', dat_filename);
            continue;
        end
        track_id = str2double(track_id_match{1}{1});
        
        track_pattern = sprintf('Tracks_%d_*.txt', track_id);
        track_files = dir(fullfile(track_folder, track_pattern));
        
        if isempty(track_files)
            track_pattern = sprintf('Tracks_%d.txt', track_id);
            track_files = dir(fullfile(track_folder, track_pattern));
        end
        
        if isempty(track_files)
            warning('找不到航迹文件：Tracks_%d_*.txt', track_id);
            % 继续处理，但不知道航迹点数
            n_track_points = -1;  % 标记为未知
        else
            % 读取第一个匹配的航迹文件
            track_data = readtable(fullfile(track_folder, track_files(1).name), ...
                'ReadVariableNames', false);
            n_track_points = height(track_data);
            fprintf('  航迹文件: %s, 航迹点数: %d\n', track_files(1).name, n_track_points);
        end
        
        % 处理原始数据文件
        filepath = fullfile(raw_data_folder, dat_filename);
        
        matrix_count = process_test_track_file(filepath, output_base, track_id, n_track_points);
        
        % 更新统计
        if matrix_count > 0
            total_processed = total_processed + 1;
            total_matrices = total_matrices + matrix_count;
            track_id_list = [track_id_list, track_id];
        end
        
    catch ME
        warning('处理文件 %s 时出错: %s', dat_filename, ME.message);
        failed_files{end+1} = dat_filename;
    end
end

%% 6. 生成处理报告
fprintf('\n\n========== 测试集处理完成 ==========\n');
fprintf('成功处理文件数: %d\n', total_processed);
fprintf('生成矩阵总数: %d\n', total_matrices);
fprintf('失败文件数: %d\n', length(failed_files));
fprintf('处理的航迹ID: %s\n', num2str(track_id_list));

% 保存处理报告
report_file = fullfile(output_base, 'test_processing_report.txt');
fid = fopen(report_file, 'w', 'n', 'UTF-8');
fprintf(fid, '测试集处理报告\n');
fprintf(fid, '==========================================\n');
fprintf(fid, '处理时间：%s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '输入目录：%s\n', raw_data_folder);
fprintf(fid, '输出目录：%s\n\n', output_base);

fprintf(fid, '处理统计：\n');
fprintf(fid, '成功处理文件数: %d\n', total_processed);
fprintf(fid, '生成矩阵总数: %d\n', total_matrices);
fprintf(fid, '失败文件数: %d\n', length(failed_files));
fprintf(fid, '处理的航迹ID: %s\n', num2str(track_id_list));

if ~isempty(failed_files)
    fprintf(fid, '\n失败文件列表:\n');
    for i = 1:length(failed_files)
        fprintf(fid, '  %s\n', failed_files{i});
    end
end

fclose(fid);
fprintf('\n报告已保存到: %s\n', report_file);

%% 处理测试集航迹文件的函数
function matrix_count = process_test_track_file(filepath, output_base, track_id, n_track_points)
    
    matrix_count = 0;
    
    % 打开文件
    fid = fopen(filepath, 'r');
    if fid == -1
        error('无法打开文件: %s', filepath);
    end
    
    % 读取所有帧数据和航迹点信息
    all_frames_rd = {};  % 存储每帧的RD矩阵
    all_point_indices = [];  % 存储每帧对应的航迹点序号
    frame_count = 0;
    
    fprintf('    读取文件: %s\n', filepath);
    
    while ~feof(fid)
        % 读取一帧数据
        [para, data_out] = funcRawDataParser(fid);
        
        if isempty(para) || isempty(data_out)
            break;
        end
        
        % MTD处理并生成RD矩阵
        rd_matrix = process_frame_to_rd(para, data_out);
        
        if ~isempty(rd_matrix)
            frame_count = frame_count + 1;
            all_frames_rd{frame_count} = rd_matrix;
            
            % 从para中提取航迹点序号
            point_idx = para.Track_No_info(2);
            all_point_indices(frame_count) = point_idx;
        end
    end
    
    fclose(fid);
    
    fprintf('      总帧数: %d\n', frame_count);
    
    % 如果没有帧数据，跳过
    if frame_count == 0
        warning('文件没有有效帧数据');
        return;
    end
    
    % 如果不知道航迹点数，从数据中推断
    if n_track_points == -1
        n_track_points = max(all_point_indices);
        fprintf('      从数据推断航迹点数: %d\n', n_track_points);
    end
    
    %按航迹点分组
    point_frame_groups = cell(n_track_points, 1);
    
    for i = 1:frame_count
        point_idx = all_point_indices(i);
        
        if point_idx >= 1 && point_idx <= n_track_points
            if isempty(point_frame_groups{point_idx})
                point_frame_groups{point_idx} = {};
            end
            point_frame_groups{point_idx}{end+1} = all_frames_rd{i};
        end
    end
    
    % === 累积处理===
    target_frames = 16;  % 目标帧数
    accumulated_groups = {};  % 存储累积组
    
    current_group_frames = {};  % 当前组的帧
    current_group_points = [];  % 当前组包含的航迹点
    
    for point_idx = 1:n_track_points
        point_frames = point_frame_groups{point_idx};
        n_frames = length(point_frames);
        
        if n_frames == 0
            continue;  % 跳过没有数据的航迹点
        end
        
        % 检查加入当前航迹点后的总帧数
        total_frames_if_added = length(current_group_frames) + n_frames;
        
        if isempty(current_group_frames)
            current_group_frames = point_frames;
            current_group_points = point_idx;
            
        elseif total_frames_if_added <= target_frames
            current_group_frames = [current_group_frames, point_frames];
            current_group_points = [current_group_points, point_idx];
            
        elseif length(current_group_frames) < target_frames && total_frames_if_added > target_frames
            if length(current_group_frames) < 8
                current_group_frames = [current_group_frames, point_frames];
                current_group_points = [current_group_points, point_idx];
            else
                accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                    'points', current_group_points, ...
                    'n_frames', length(current_group_frames));
                
                current_group_frames = point_frames;
                current_group_points = point_idx;
            end
            
        else
            accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                'points', current_group_points, ...
                'n_frames', length(current_group_frames));
            
            current_group_frames = point_frames;
            current_group_points = point_idx;
        end
        
        if length(current_group_frames) >= target_frames
            accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                'points', current_group_points, ...
                'n_frames', length(current_group_frames));
            
            current_group_frames = {};
            current_group_points = [];
        end
    end
    
    % 处理最后一组
    if ~isempty(current_group_frames)
        accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
            'points', current_group_points, ...
            'n_frames', length(current_group_frames));
    end
    
    % 处理每个累积组
    for group_idx = 1:length(accumulated_groups)
        group = accumulated_groups{group_idx};
        
        % 创建时序拼接矩阵
        combined_matrix = create_temporal_matrix_fixed(group.frames);
        
        if ~isempty(combined_matrix)
            %生成测试集的文件名不包含标签
            save_filename = sprintf('Track%d_Group%03d_Points%d-%d.mat', ...
                track_id, group_idx, ...
                min(group.points), max(group.points));
            
            % 保存到测试集目录
            save_path = fullfile(output_base, 'test',  save_filename);
            
            % 保存测试集不包含标签
            data = combined_matrix;  % 32×64矩阵
            
            % 保存元数据
            metadata = struct();
            metadata.track_id = track_id;
            metadata.group_idx = group_idx;
            metadata.included_points = group.points;
            metadata.n_frames = group.n_frames;
            metadata.original_frame_count = frame_count;
            metadata.target_frames = target_frames;
            
            save(save_path, 'data', 'metadata'); 
            
            matrix_count = matrix_count + 1;
        end
    end
    
    fprintf('      生成 %d 个矩阵（从%d个组）\n', matrix_count, length(accumulated_groups));
end

% create_temporal_matrix_fixed 函数
function combined_matrix = create_temporal_matrix_fixed(frames_rd)
    n_frames = length(frames_rd);
    
    % 步骤1：距离维压缩
    compressed_vectors = zeros(n_frames, 64);
    
    for i = 1:n_frames
        rd_matrix = frames_rd{i};
        compressed_vectors(i, :) = sum(abs(rd_matrix), 1);
    end
    
    % 步骤2：调整为32×64输出
    combined_matrix = zeros(32, 64);
    
    if n_frames == 32
        combined_matrix = compressed_vectors;
        
    elseif n_frames < 32
        if n_frames == 1
            center_row = 16;
            sigma = 5;
            for row = 1:32
                weight = exp(-(row - center_row)^2 / (2 * sigma^2));
                combined_matrix(row, :) = compressed_vectors(1, :) * weight;
            end
            
        elseif n_frames <= 20
            old_indices = 1:n_frames;
            new_indices = linspace(1, n_frames, 32);
            
            for col = 1:64
                if n_frames >= 4
                    combined_matrix(:, col) = interp1(old_indices, ...
                        compressed_vectors(:, col), new_indices, 'spline');
                else
                    combined_matrix(:, col) = interp1(old_indices, ...
                        compressed_vectors(:, col), new_indices, 'linear');
                end
            end
            
        else
            old_indices = 1:n_frames;
            new_indices = linspace(1, n_frames, 32);
            
            for col = 1:64
                combined_matrix(:, col) = interp1(old_indices, ...
                    compressed_vectors(:, col), new_indices, 'linear');
            end
        end
        
    else
        sample_indices = round(linspace(1, n_frames, 32));
        combined_matrix = compressed_vectors(sample_indices, :);
    end
end

% process_frame_to_rd 函数
function rd_matrix = process_frame_to_rd(para, data_out)
    rd_matrix = [];
    
    try
        % 计算速度轴
        c = 3e8;
        delta_Vr = c / (2 * size(data_out, 2) * para.PRT * para.Freq);
        
        delta_Vr0 = 0.1; % m/s
        VelocityNum = size(data_out, 2) * delta_Vr/delta_Vr0;
        VelocityNum = 2^(ceil(log(VelocityNum)/log(2)));
        
        % MTD处理
        MTD_win = taylorwin(size(data_out, 2), [], -30);
        coef_MTD_2D = repmat(MTD_win, [1, size(data_out, 1)]);
        coef_MTD_2D = permute(coef_MTD_2D, [2, 1]);
        
        data_out = data_out - mean(data_out, 2);
        data_proc_MTD_win_out = data_out .* coef_MTD_2D;
        data_proc_MTD_result = fftshift(fft(data_proc_MTD_win_out, VelocityNum, 2), 2);
        
        % 速度轴
        Vr = (-VelocityNum/2 : VelocityNum/2-1) * delta_Vr * size(data_out, 2) / VelocityNum;
        
        % 提取固定速度范围
        Vr_target = [-30, 30];  % 目标速度范围 (m/s)
        
        [~, col_zero] = min(abs(Vr));  % 找到零速度位置
        [~, col_start] = min(abs(Vr - Vr_target(1)));  % -30 m/s 对应的索引
        [~, col_end] = min(abs(Vr - Vr_target(2)));    % +30 m/s 对应的索引
        
        % 确保提取64列
        n_cols = col_end - col_start + 1;
        
        if n_cols > 64
            col_indices = round(linspace(col_start, col_end, 64));
        elseif n_cols < 64
            col_start = max(1, col_zero - 32);
            col_end = min(size(data_proc_MTD_result, 2), col_zero + 31);
            
            if col_end - col_start + 1 < 64
                if col_start == 1
                    col_end = min(size(data_proc_MTD_result, 2), 64);
                else
                    col_start = max(1, col_end - 63);
                end
            end
            col_indices = col_start:col_end;
        else
            col_indices = col_start:col_end;
        end
        
        % 提取RD矩阵
        rd_matrix = abs(data_proc_MTD_result(:, col_indices));
        
    catch ME
        warning('MTD处理失败: %s', ME.message);
        rd_matrix = [];
    end
end

% funcRawDataParser 函数
function [para, data_out] = funcRawDataParser(fid)
    para = [];
    data_out = [];
    
    frame_head = hex2dec('FA55FA55');
    frame_end = hex2dec('55FA55FA');
    
    % 查找帧头
    head_find = fread(fid, 1, 'uint32');
    if isempty(head_find)
        return;
    end
    
    while head_find ~= frame_head && ~feof(fid)
        fseek(fid, -3, 'cof');
        head_find = fread(fid, 1, 'uint32');
        if feof(fid)
            return;
        end
    end
    
    % 读取帧长度
    frame_data_length = fread(fid, 1, 'uint32');
    if isempty(frame_data_length)
        return;
    end
    frame_data_length = frame_data_length * 4;
    
    % 检查帧尾
    fseek(fid, frame_data_length - 12, 'cof');
    end_find = fread(fid, 1, 'uint32');
    
    % 验证帧完整性
    while (head_find ~= frame_head) || (end_find ~= frame_end)
        fseek(fid, -frame_data_length + 1, 'cof');
        
        head_find = fread(fid, 1, 'uint32');
        if isempty(head_find)
            return;
        end
        
        frame_data_length = fread(fid, 1, 'uint32');
        if isempty(frame_data_length)
            return;
        end
        frame_data_length = frame_data_length * 4;
        
        fseek(fid, frame_data_length - 8, 'cof');
        end_find = fread(fid, 1, 'uint32');
        
        if feof(fid) && (head_find ~= frame_head || end_find ~= frame_end)
            return;
        end
    end
    
    % 回到数据开始位置
    fseek(fid, -frame_data_length + 4, 'cof');
    
    % 读取数据头
    data_temp1 = fread(fid, 3, 'uint32');
    para.E_scan_Az = data_temp1(2) * 0.01;
    pointNum_in_bowei = data_temp1(3);
    
    % 读取参数
    data_temp = fread(fid, pointNum_in_bowei * 4 + 5, 'uint32');
    para.Track_No_info = data_temp(1:pointNum_in_bowei * 4);
    para.Freq = data_temp(pointNum_in_bowei * 4 + 1) * 1e6;
    para.CPIcount = data_temp(pointNum_in_bowei * 4 + 2);
    para.PRTnum = data_temp(pointNum_in_bowei * 4 + 3);
    para.PRT = data_temp(pointNum_in_bowei * 4 + 4) * 0.0125e-6;
    para.data_length = data_temp(pointNum_in_bowei * 4 + 5);
    
    % 读取IQ数据
    data_out_temp = fread(fid, para.PRTnum * 31 * 2, 'float');
    if isempty(data_out_temp) || length(data_out_temp) < para.PRTnum * 31 * 2
        para = [];
        data_out = [];
        return;
    end
    
    % 组织复数数据
    data_out_real = data_out_temp(1:2:end);
    data_out_imag = data_out_temp(2:2:end);
    data_out_complex = data_out_real + 1i * data_out_imag;
    data_out = reshape(data_out_complex, 31, para.PRTnum);
    
    % 距离维FFT处理
    Dwin = taylorwin(31, 4, -35);
    data_out = data_out .* Dwin;
    data_out = fftshift(fft(data_out, 64, 1), 1);
    
    % 跳过帧尾
    fseek(fid, 4, 'cof');
end