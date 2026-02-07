import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from drsncww import rsnet34
from data_loader_new import GesDataLoaderNew
import re
import chardet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrackResultWriter:
    """
    航迹识别结果写入器
    功能：将识别结果写入航迹文件的最后一列
    """

    def __init__(self, track_folder_path, output_folder_path):
        self.track_folder = Path(track_folder_path)
        self.output_folder = Path(output_folder_path)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # 类别映射：模型输出 -> 航迹文件标签
        self.class_mapping = {
            0: 1,  # 1_Lightweight_UAV -> 标签1
            1: 2,  # 2_Small_UAV -> 标签2
            2: 3,  # 3Birds -> 标签3
            3: 4,  # 4Fly_ball -> 标签4
            # 移除 -1: 0 的映射，未识别时将继承上一个点的结果
        }

    def detect_encoding(self, file_path):
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding']
        except:
            return 'utf-8'

    def read_track_file(self, track_file_path):
        """
        读取航迹文件，自动处理编码问题
        返回: (lines, encoding)
        """
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']

        # 尝试自动检测编码
        try:
            detected_encoding = self.detect_encoding(track_file_path)
            if detected_encoding:
                encodings.insert(0, detected_encoding)
        except:
            pass

        for encoding in encodings:
            try:
                with open(track_file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                return lines, encoding
            except UnicodeDecodeError:
                continue

        # 最后尝试忽略错误
        with open(track_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines(), 'utf-8'

    def extract_points_info(self, filename):
        """
        从矩阵文件名提取Points信息
        例如: Track1_Group001_Points1-2.mat -> (1, 2)
        例如: Track1278_Group001_Points1-4.mat -> (1, 4)
        """
        match = re.search(r'Points(\d+)-(\d+)', filename)
        if match:
            start_point = int(match.group(1))
            end_point = int(match.group(2))
            return start_point, end_point
        return None, None

    def extract_track_id(self, filename):
        """
        从文件名提取航迹ID
        例如: Track1_Group001_Points1-2.mat -> 1
        """
        match = re.search(r'Track(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None

    def find_track_file(self, track_id):
        """
        根据track_id查找对应的航迹文件
        支持多种命名格式：Tracks_1_1_21.txt, Tracks_1_20.txt等
        """
        possible_patterns = [
            f"Tracks_{track_id}_*.txt",
            f"Tracks_{track_id}.txt",
            f"Track_{track_id}_*.txt",
            f"Track_{track_id}.txt"
        ]

        for pattern in possible_patterns:
            track_files = list(self.track_folder.glob(pattern))
            if track_files:
                return track_files[0]  # 返回第一个匹配的文件

        return None

    def write_track_results(self, inference_results):
        """
        将推理结果写入航迹文件

        参数:
            inference_results: 推理结果列表，包含filename和prediction等信息
        """
        print("=" * 60)
        print("开始写入航迹识别结果")
        print("=" * 60)

        # 按航迹ID分组推理结果
        track_predictions = {}

        for result in inference_results:
            filename = result['filename']
            track_id = self.extract_track_id(filename)
            start_point, end_point = self.extract_points_info(filename)

            if track_id is None or start_point is None:
                print(f"警告：无法解析文件名 {filename}")
                continue

            if track_id not in track_predictions:
                track_predictions[track_id] = {}

            # 只有成功识别的结果才记录，未识别的(-1)不记录
            prediction = result['prediction']
            if prediction != -1:  # 只处理成功识别的结果
                track_label = self.class_mapping.get(prediction, 0)

                # 记录这个预测应该写入的行号（end_point对应的行）
                track_predictions[track_id][end_point] = {
                    'label': track_label,
                    'filename': filename,
                    'confidence': result['confidence'],
                    'class_name': result['class_name'],
                    'points_range': f"{start_point}-{end_point}"
                }
            else:
                print(f"跳过未识别结果: {filename} (将继承上一点的标签)")

        print(f"处理 {len(track_predictions)} 个航迹的识别结果")

        # 处理每个航迹文件
        processed_count = 0
        for track_id, predictions in track_predictions.items():
            track_file = self.find_track_file(track_id)

            if track_file is None:
                print(f"警告：找不到航迹ID {track_id} 对应的航迹文件")
                continue

            try:
                self.process_single_track_file(track_file, track_id, predictions)
                processed_count += 1
            except Exception as e:
                print(f"处理航迹文件 {track_file.name} 时出错: {e}")

        print(f"\n成功处理 {processed_count} 个航迹文件")
        print(f"结果保存在: {self.output_folder}")

    def process_single_track_file(self, track_file, track_id, predictions):
        """
        处理单个航迹文件，写入识别结果

        修改后的逻辑：
        1. 航迹开始时：所有行都是0
        2. 第一次识别结果出来后：从该行开始延续该结果
        3. 新的识别结果出来后：从该行开始更新为新结果
        4. 未识别的情况：继续保持上一个点的识别结果

        参数:
            track_file: 航迹文件路径
            track_id: 航迹ID
            predictions: 该航迹的所有预测结果 {点号: 预测信息} (只包含成功识别的结果)
        """
        print(f"\n处理航迹文件: {track_file.name}")

        # 读取原始航迹文件
        lines, encoding = self.read_track_file(track_file)
        print(f"  检测到编码: {encoding}")
        print(f"  原始行数: {len(lines)}")

        if not lines:
            print("  警告：文件为空")
            return

        # 准备输出行
        output_lines = []

        # 处理表头
        header = lines[0].strip()
        if not header.endswith(',识别结果'):
            header += ',识别结果'
        output_lines.append(header + '\n')

        # 处理数据行
        data_lines = lines[1:]  # 去掉表头
        max_point = len(data_lines)  # 航迹点总数

        print(f"  航迹点总数: {max_point}")
        print(f"  预测结果覆盖的点: {sorted(predictions.keys())}")

        # 初始化所有行的识别结果为0（未开始识别）
        recognition_results = [0] * max_point

        # 按时间顺序获取所有识别结果更新点（只包含成功识别的点）
        update_points = sorted(predictions.keys())

        if not update_points:
            print("  注意：该航迹没有成功识别的结果，所有点保持标签0")
        else:
            # 逐个处理每个识别结果更新点
            current_label = 0  # 当前标签，初始为0

            for i, update_point in enumerate(update_points):
                # 获取新的识别结果
                new_label = predictions[update_point]['label']
                pred_info = predictions[update_point]

                print(f"    点 {update_point}: 更新识别结果为 {new_label} ({pred_info['class_name']}) "
                      f"[{pred_info['points_range']}] 置信度:{pred_info['confidence']:.3f}")

                # 确定这个新结果的作用范围
                start_idx = update_point - 1  # 转换为0-based索引

                # 结束索引：到下一个更新点之前，或者到文件末尾
                if i + 1 < len(update_points):
                    end_idx = update_points[i + 1] - 1  # 到下一个更新点之前
                else:
                    end_idx = max_point  # 到文件末尾

                # 更新指定范围内的所有行
                for j in range(start_idx, min(end_idx, max_point)):
                    recognition_results[j] = new_label

                current_label = new_label
                print(f"      → 第{update_point}行到第{min(end_idx, max_point)}行设置为: {new_label}")

        # 生成输出行
        for i, line in enumerate(data_lines):
            line = line.strip()
            if line:
                # 添加识别结果到行末
                output_line = f"{line},{recognition_results[i]}\n"
                output_lines.append(output_line)

        # 写入输出文件
        output_file = self.output_folder / track_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)

        print(f"  ✓ 已保存: {output_file}")

        # 显示识别结果统计
        result_counts = {}
        for result in recognition_results:
            result_counts[result] = result_counts.get(result, 0) + 1

        print(f"  识别结果统计: {result_counts}")

        # 显示详细的行级别结果（前10行和后10行）
        print(f"  前10行结果: {recognition_results[:10]}")
        if max_point > 10:
            print(f"  后10行结果: {recognition_results[-10:]}")


def main_with_track_writing():
    """
    主函数：推理 + 写入航迹文件
    """
    print("=" * 60)
    print("航迹识别系统 - 推理并写入结果")
    print("=" * 60)

    # ===== 第一步：推理 =====
    print("\n第一步：执行推理...")

    # 1. 数据加载
    test_path = './dataset/preprocess_data/test'

    try:
        test_dataset = GesDataLoaderNew(test_path, data_rows=32, data_cols=64, val=True, test_mode=True)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    print(f"数据集大小: {len(test_dataset)}")

    # 2. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 模型加载
    try:
        net = rsnet34()
        s_path = './checkpoint/ckpt_best_10_94.38.pth'

        checkpoint = torch.load(s_path, map_location=device, weights_only=False)

        if 'net_weight' in checkpoint:
            net.load_state_dict(checkpoint['net_weight'])
            print("✓ 使用 'net_weight' 加载模型")
        else:
            net.load_state_dict(checkpoint)
            print("✓ 直接加载模型")

        net.to(device)
        net.eval()

    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 4. 4类目标定义
    target_classes = {
        0: "1_Lightweight_UAV",
        1: "2_Small_UAV",
        2: "3Birds",
        3: "4Fly_ball"
    }

    print(f"识别目标类别: {list(target_classes.values())}")

    # 5. 推理
    conf_threshold = 0.5
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0,
                                 drop_last=False, shuffle=False)

    results = []

    print(f"\n开始推理...")

    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            # 数据处理
            if len(data) == 3:
                input_data = data[0].type(torch.FloatTensor).to(device)
                filename = data[2][0] if isinstance(data[2], (list, tuple)) else data[2]
            else:
                input_data = data[0].type(torch.FloatTensor).to(device)
                filename = data[1][0] if isinstance(data[1], (list, tuple)) else data[1]

            print(f"样本 {step + 1:3d}: {filename}")

            # 模型推理
            outputs = net(input_data)
            probs_6class = torch.softmax(outputs, dim=1)

            # 只取前4类
            probs_4class = probs_6class[:, :4]
            probs_4class_norm = probs_4class / probs_4class.sum(dim=1, keepdim=True)

            predicted_class = torch.argmax(probs_4class_norm, dim=1)
            confidence = torch.max(probs_4class_norm, dim=1)[0]

            # 应用置信度阈值
            if confidence.item() >= conf_threshold:
                final_prediction = predicted_class.item()
                final_class_name = target_classes[final_prediction]
                status = "可信"
            else:
                final_prediction = -1
                final_class_name = "未识别"
                status = "低置信度"

            print(f"  预测: {final_prediction} ({final_class_name}) - {status}")

            # 保存结果
            result = {
                'filename': filename,
                'prediction': final_prediction,
                'class_name': final_class_name,
                'confidence': confidence.item(),
                'status': status
            }
            results.append(result)

    # ===== 第二步：写入航迹文件 =====
    print(f"\n第二步：写入航迹识别结果...")

    # 航迹文件路径配置
    track_folder = "../../database/挑战杯_揭榜挂帅_CQ-08赛题_数据集/test/航迹"  # 原始航迹文件夹
    output_folder = "../../output/航迹识别结果"  # 输出文件夹

    # 检查路径是否存在
    if not Path(track_folder).exists():
        # 尝试其他可能的路径
        alternative_paths = [
            "./航迹",
            "../航迹",
            "../../../../database/挑战杯_揭榜挂帅_CQ-08赛题_数据集/CQ-08中国航天科工二院二十三所-低空监视雷达目标智能识别技术研究数据集/航迹"
        ]

        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                track_folder = alt_path
                break
        else:
            print(f"错误：找不到航迹文件夹")
            print(f"请确保以下路径之一存在：")
            print(f"  - {track_folder}")
            for path in alternative_paths:
                print(f"  - {path}")
            return

    # 创建写入器并执行写入
    writer = TrackResultWriter(track_folder, output_folder)
    writer.write_track_results(results)

    # ===== 第三步：结果统计 =====
    print(f"\n第三步：结果统计...")

    # 统计推理结果
    pred_counts = {}
    for result in results:
        pred = result['prediction']
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

    print("矩阵级别预测分布:")
    for pred, count in sorted(pred_counts.items()):
        if pred == -1:
            print(f"  未识别: {count}")
        else:
            print(f"  类别 {pred} ({target_classes[pred]}): {count}")

    print(f"\n全部完成！")
    print(f"航迹识别结果已保存到: {output_folder}")


if __name__ == "__main__":
    main_with_track_writing()