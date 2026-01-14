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

% 创建输出目录结构
for split = {'train', 'val', 'test'}
    for label = 0:5  % 神经网络使用0-5标签
        output_dir = fullfile(output_base, split{1}, num2str(label));
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
    end
end

%% 2. 定义类别名称
class_names = {
    '轻型旋翼无人机',     % Label 1 -> 0
    '小型旋翼无人机',     % Label 2 -> 1  
    '鸟类',             % Label 3 -> 2
    '空飘球',           % Label 4 -> 3
    '杂波',             % Label 5 -> 4
    '其它未识别目标'     % Label 6 -> 5
};

%% 3. 设置参数
rng(42);  % 保证可重复性
Fs = 20e6;
c = 3e8;
delta_R = c/2/Fs;

%% 4. 获取所有.dat文件
dat_files = dir(fullfile(raw_data_folder, '*_Label_*.dat'));
fprintf('找到 %d 个.dat文件\n', length(dat_files));

%% 5. 初始化统计变量
total_processed = 0;
total_matrices = 0;
label_counts = zeros(6, 1);
failed_files = {};
split_counts = struct('train', 0, 'val', 0, 'test', 0);

% 按标签分组
label_groups = cell(6, 1);
for i = 1:length(dat_files)
    tokens = regexp(dat_files(i).name, '_Label_(\d+)', 'tokens');
    if ~isempty(tokens)
        original_label = str2double(tokens{1}{1});
        if original_label >= 1 && original_label <= 6
            label_groups{original_label}{end+1} = i;
        end
    end
end

%% 6. 处理每个类别
for label_idx = 1:6
    files_in_class = label_groups{label_idx};
    n_files = length(files_in_class);
    
    if n_files == 0
        fprintf('警告：类别 %d (%s) 没有数据文件\n', label_idx, class_names{label_idx});
        continue;
    end
    
    fprintf('\n处理类别 %d (%s): %d 个文件\n', label_idx, class_names{label_idx}, n_files);
    
    % 随机打乱该类别的文件
    rand_idx = randperm(n_files);
    
    % 分配到train/val/test (60/20/20)
    train_end = round(n_files * 0.6);
    val_end = round(n_files * 0.8);
    
    % 创建进度条
    h = waitbar(0, sprintf('处理类别 %d (%s)...', label_idx, class_names{label_idx}));
    
    for i = 1:n_files
        file_idx = files_in_class{rand_idx(i)};
        
        try
            % 提取文件信息
            tokens = regexp(dat_files(file_idx).name, '(\d+)_Label_(\d+)', 'tokens');
            track_id = str2double(tokens{1}{1});
            original_label = str2double(tokens{1}{2});
            nn_label = original_label - 1;  % 转换为0-5
            
            % 决定数据集归属
            if i <= train_end
                split = 'train';
            elseif i <= val_end
                split = 'val';
            else
                split = 'test';
            end
            
            % 读取航迹文件获取航迹点数
            track_file = dir(fullfile(track_folder, sprintf('Tracks_%d_%d_*.txt', track_id, original_label)));
            if isempty(track_file)
                warning('找不到航迹文件：Track_%d_%d', track_id, original_label);
                continue;
            end
            
            track_data = readtable(fullfile(track_folder, track_file(1).name), 'ReadVariableNames', false);
            n_track_points = height(track_data);
            
            % 处理该文件的每个航迹点
            filepath = fullfile(raw_data_folder, dat_files(file_idx).name);
            matrix_count = process_track_file(filepath, output_base, split, nn_label, track_id, n_track_points);
            
            % 更新统计
            if matrix_count > 0
                total_processed = total_processed + 1;
                total_matrices = total_matrices + matrix_count;
                split_counts.(split) = split_counts.(split) + matrix_count;
                label_counts(original_label) = label_counts(original_label) + matrix_count;
            end
            
            % 更新进度条
            waitbar(i/n_files, h, sprintf('处理类别 %d (%s): %d/%d 文件', ...
                label_idx, class_names{label_idx}, i, n_files));
            
        catch ME
            warning('处理文件 %s 时出错: %s', dat_files(file_idx).name, ME.message);
            failed_files{end+1} = dat_files(file_idx).name;
        end
    end
    
    close(h);
end

%% 7. 生成报告
fprintf('\n\n========== 处理完成 ==========\n');
fprintf('成功处理文件数: %d\n', total_processed);
fprintf('生成矩阵总数: %d\n', total_matrices);
fprintf('失败文件数: %d\n', length(failed_files));

fprintf('\n各类别生成矩阵数量:\n');
for i = 1:6
    fprintf('  %s (标签%d->%d): %d 个矩阵\n', class_names{i}, i, i-1, label_counts(i));
end

fprintf('\n数据集划分:\n');
fprintf('  训练集: %d 个矩阵\n', split_counts.train);
fprintf('  验证集: %d 个矩阵\n', split_counts.val);
fprintf('  测试集: %d 个矩阵\n', split_counts.test);

% 保存处理报告
report_file = fullfile(output_base, 'processing_report.txt');
fid = fopen(report_file, 'w', 'n', 'UTF-8');
fprintf(fid, '基于航迹点的时序RD图处理报告\n');
fprintf(fid, '==========================================\n');
fprintf(fid, '处理时间：%s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '输入目录：%s\n', raw_data_folder);
fprintf(fid, '输出目录：%s\n\n', output_base);

fprintf(fid, '处理统计：\n');
fprintf(fid, '成功处理文件数: %d\n', total_processed);
fprintf(fid, '生成矩阵总数: %d\n', total_matrices);
fprintf(fid, '失败文件数: %d\n\n', length(failed_files));

fprintf(fid, '各类别生成矩阵数量:\n');
for i = 1:6
    fprintf(fid, '  %s (标签%d->%d): %d 个矩阵\n', class_names{i}, i, i-1, label_counts(i));
end

fprintf(fid, '\n数据集划分:\n');
fprintf(fid, '  训练集: %d 个矩阵\n', split_counts.train);
fprintf(fid, '  验证集: %d 个矩阵\n', split_counts.val);
fprintf(fid, '  测试集: %d 个矩阵\n', split_counts.test);

if ~isempty(failed_files)
    fprintf(fid, '\n失败文件列表:\n');
    for i = 1:length(failed_files)
        fprintf(fid, '  %s\n', failed_files{i});
    end
end

fclose(fid);
fprintf('\n报告已保存到: %s\n', report_file);

%% 处理单个航迹文件的函数(要累计16帧以后在处理)
function matrix_count = process_track_file(filepath, output_base, split, nn_label, track_id, n_track_points)
    
    matrix_count = 0;
    
    % 打开文件
    fid = fopen(filepath, 'r');
    if fid == -1
        error('无法打开文件: %s', filepath);
    end
    
    % 首先读取所有帧数据和航迹点信息
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
                %%==========调试矩阵的距离维度求和是否正确===========
%     imagesc(rd_matrix);
%     colorbar;    
%     % 保存当前图像为 PNG 文件
%     saveas(gcf, ['D:\\E\\sum1\\matrix_''.png']);
%%==========调试结束=============================
        
        if ~isempty(rd_matrix)
            frame_count = frame_count + 1;
            all_frames_rd{frame_count} = rd_matrix;
            
            % 从para中提取航迹点序号
            point_idx = para.Track_No_info(2);
            all_point_indices(frame_count) = point_idx;
        end
    end
    
    fclose(fid);
    
    fprintf('      总帧数: %d, 航迹点数: %d\n', frame_count, n_track_points);
    
    % 如果没有帧数据，跳过
    if frame_count == 0
        warning('文件没有有效帧数据');
        return;
    end
    
    % === 新策略：累积16帧的处理方式 ===
    
    % 首先按航迹点分组
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
    
    % 显示原始分布
    fprintf('      原始航迹点帧数分布：\n');
    for point_idx = 1:min(10, n_track_points)
        n_frames_for_point = length(point_frame_groups{point_idx});
        if n_frames_for_point > 0
            fprintf('        航迹点%d: %d帧\n', point_idx, n_frames_for_point);
        end
    end
    
    % === 核心修改：累积处理策略 ===
    target_frames = 16;  % 目标帧数
    accumulated_frames = {};  % 累积的帧
    accumulated_points = [];  % 记录哪些航迹点被累积
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
        
        % 决策逻辑
        if isempty(current_group_frames)
            % 第一个航迹点，直接加入
            current_group_frames = point_frames;
            current_group_points = point_idx;
            
        elseif total_frames_if_added <= target_frames
            % 加入后不超过16帧，直接加入
            current_group_frames = [current_group_frames, point_frames];
            current_group_points = [current_group_points, point_idx];
            
        elseif length(current_group_frames) < target_frames && total_frames_if_added > target_frames
            % 当前不足16帧，但加入后超过16帧
            % 策略：如果当前帧数太少小于8，则全部加入；否则保存当前组，开始新组
            
            if length(current_group_frames) < 8
                % 当前太少，全部加入
                current_group_frames = [current_group_frames, point_frames];
                current_group_points = [current_group_points, point_idx];
            else
                % 保存当前组，开始新组
                accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                    'points', current_group_points, ...
                    'n_frames', length(current_group_frames));
                
                % 开始新组
                current_group_frames = point_frames;
                current_group_points = point_idx;
            end
            
        else
            % 当前已经>=16帧，保存并开始新组
            accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                'points', current_group_points, ...
                'n_frames', length(current_group_frames));
            
            % 开始新组
            current_group_frames = point_frames;
            current_group_points = point_idx;
        end
        
        % 如果当前组达到目标帧数，保存它
        if length(current_group_frames) >= target_frames
            accumulated_groups{end+1} = struct('frames', {current_group_frames}, ...
                'points', current_group_points, ...
                'n_frames', length(current_group_frames));
            
            % 重置
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
    
%     % 显示累积结果
%     fprintf('\n      累积分组结果（目标%d帧）：\n', target_frames);
%     for g = 1:length(accumulated_groups)
%         group = accumulated_groups{g};
%         fprintf('        组%d: 包含航迹点[%s], 共%d帧\n', ...
%             g, num2str(group.points), group.n_frames);
%     end
    
    % 处理每个累积组
    for group_idx = 1:length(accumulated_groups)
        group = accumulated_groups{group_idx};
        
        % 创建时序拼接矩阵
        combined_matrix = create_temporal_matrix_fixed(group.frames);
        
        if ~isempty(combined_matrix)
            % 生成保存文件名，使用组的第一个航迹点作为标识
            first_point = group.points(1);
            save_filename = sprintf('Track%d_Label%d_Group%03d_Points%d-%d.mat', ...
                track_id, nn_label, group_idx, ...
                min(group.points), max(group.points));
            save_path = fullfile(output_base, split, num2str(nn_label), save_filename);
            
            % 保存数据和元信息
            data = combined_matrix;  % 32×64矩阵
            label = nn_label;  % 0-5的标签
            
            % 保存元数据
            metadata = struct();
            metadata.track_id = track_id;
            metadata.group_idx = group_idx;
            metadata.included_points = group.points;
            metadata.n_frames = group.n_frames;
            metadata.original_frame_count = frame_count;
            metadata.target_frames = target_frames;
            
            save(save_path, 'data', 'label', 'metadata');
            
            matrix_count = matrix_count + 1;
        end
    end
    
    fprintf('      生成 %d 个矩阵（从%d个组）\n', matrix_count, length(accumulated_groups));
end

%% 时序拼接函数（16帧左右的数据优化）
function combined_matrix = create_temporal_matrix_fixed(frames_rd)
    n_frames = length(frames_rd);
    compressed_vectors = zeros(n_frames, 64);
    for i = 1:n_frames
        compressed_vectors(i, :) = sum(abs(frames_rd{i}), 1);
    end
    if n_frames == 32
        combined_matrix = compressed_vectors;
    elseif n_frames == 1
        center_row = 16;
        sigma = 5;
        for row = 1:32
            weight = exp(-(row - center_row)^2 / (2 * sigma^2));
            combined_matrix(row, :) = compressed_vectors(1, :) * weight;
        end
    else
%         fprintf('成功interp2');
        [X, Y] = ndgrid(1:n_frames, 1:64);
        [Xq, Yq] = ndgrid(linspace(1, n_frames, 32), 1:64);
        if(n_frames >= 4)
            combined_matrix = interp2(Y, X, compressed_vectors, Yq, Xq, 'spline');
        else
            combined_matrix = interp2(Y, X, compressed_vectors, Yq, Xq, 'linear');
        end

        
    end
end


%% MTD处理函数 - 提取-30-30的速度范围：与RdMapParser相似的做法
function rd_matrix = process_frame_to_rd(para, data_out)
    rd_matrix = [];
    
    try
        % 计算速度轴
        c = 3e8;
        delta_Vr = c / (2 * size(data_out, 2) * para.PRT * para.Freq);
        
%         delta_Vr0 = 0.1; % m/s
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
        
        % ==== 关键修改：提取固定速度范围 ====
        % 定义固定的速度范围
        Vr_target = [-30, 30];  % 目标速度范围 (m/s)
        
        % 找到零速度位置（在所有情况下都需要）
        [~, col_zero] = min(abs(Vr));  % 找到零速度位置
        
        % 找到对应的列索引
        [~, col_start] = min(abs(Vr - Vr_target(1)));  % -30 m/s 对应的索引
        [~, col_end] = min(abs(Vr - Vr_target(2)));    % +30 m/s 对应的索引
        
        % 确保提取64列（如果速度范围对应的列数不是64，进行调整）
        n_cols = col_end - col_start + 1;
        
        if n_cols > 64
            % 如果列数超过64，均匀采样64列
            col_indices = round(linspace(col_start, col_end, 64));
        elseif n_cols < 64
            % 如果列数不足64，以零速度为中心扩展
            col_start = max(1, col_zero - 32);
            col_end = min(size(data_proc_MTD_result, 2), col_zero + 31);
            
            % 确保正好64列
            if col_end - col_start + 1 < 64
                if col_start == 1
                    col_end = min(size(data_proc_MTD_result, 2), 64);
                else
                    col_start = max(1, col_end - 63);
                end
            end
            col_indices = col_start:col_end;
        else
            % 正好64列
            col_indices = col_start:col_end;
        end
        
        % ==== 提取固定的距离范围 ====
        % 提取RD矩阵
        rd_matrix = abs(data_proc_MTD_result(:, col_indices));
        
%         % 如果距离维不是64，调整到64
%         if size(rd_matrix, 1) ~= 64
%             if size(rd_matrix, 1) > 64
%                 % 取中心64行
%                 center_row = floor(size(rd_matrix, 1) / 2);
%                 row_start = max(1, center_row - 32);
%                 row_end = min(size(rd_matrix, 1), row_start + 63);
%                 rd_matrix = rd_matrix(row_start:row_end, :);
%             else
%                 % 补零到64行
%                 temp = zeros(64, size(rd_matrix, 2));
%                 temp(1:size(rd_matrix, 1), :) = rd_matrix;
%                 rd_matrix = temp;
%             end
%         end
%         
%         % 确保输出是32×64（根据您的需求）
%         if size(rd_matrix, 1) == 64
%             % 隔行采样到32行
%             rd_matrix = rd_matrix(1:2:end, :);  
%         end
        
        % 调试输出 - 找到col_zero在col_indices中的位置
        zero_idx_in_result = find(col_indices == col_zero, 1);
        if isempty(zero_idx_in_result)
            % 如果零速度不在提取的范围内，找最接近的
            [~, zero_idx_in_result] = min(abs(col_indices - col_zero));
        end
        
%         fprintf('提取的速度范围: %.1f 到 %.1f m/s\n', Vr(col_indices(1)), Vr(col_indices(end)));
%         fprintf('零速度在第 %d 列（共%d列）\n', zero_idx_in_result, length(col_indices));
%         fprintf('输出RD矩阵大小: [%d × %d]\n', size(rd_matrix));
        
    catch ME
        warning('MTD处理失败: %s', ME.message);
        rd_matrix = [];
    end
end




%% 原始数据解析函数
function [para, data_out] = funcRawDataParser(fid)
    % 初始化
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
    % Dwin = ones(31,1);  % 可选：不加窗
    data_out = data_out .* Dwin;
    data_out = fftshift(fft(data_out, 64, 1), 1);
    
    % 跳过帧尾
    fseek(fid, 4, 'cof');
end

