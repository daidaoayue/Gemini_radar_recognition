import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from drsncww import rsnet34
from data_loader_new import GesDataLoaderNew

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def inference_main():
    """
    修复后的4类预测推理主函数 - 使用原有数据加载器
    """
    print("=" * 60)
    print("4类目标识别推理程序（修复版）")
    print("=" * 60)

    # 1. 数据加载 - 使用原有的数据加载器但设置为测试模式
    test_path = './dataset/preprocess_data/test'

    try:
        # 使用原有的数据加载器，确保数据预处理正确
        test_dataset = GesDataLoaderNew(test_path, data_rows=32, data_cols=64, val=True, test_mode=True)
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("尝试使用不同的数据加载方式...")

        # 如果上面失败，尝试其他方式
        try:
            from data_loader_new import TestDataLoader
            test_dataset = TestDataLoader(test_path, data_rows=32, data_cols=64)
        except:
            print("所有数据加载方式都失败")
            return

    print(f"\n数据集大小: {len(test_dataset)}")

    # 2. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 模型加载
    try:
        net = rsnet34()
        s_path = './checkpoint/ckpt_best_91_94.45.pth'

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
        0: "1_Lightweight_UAV",  # 轻型旋翼无人机
        1: "2_Small_UAV",  # 小型旋翼无人机
        2: "3Birds",  # 鸟类
        3: "4Fly_ball"  # 空飘球
    }

    print(f"识别目标类别: {list(target_classes.values())}")

    # 5. 推理参数
    conf_threshold = 0.5  # 置信度阈值
    print(f"置信度阈值: {conf_threshold}")

    # 6. 推理循环
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0,
                                 drop_last=False, shuffle=False)

    results = []  # 存储所有推理结果

    print(f"\n开始推理...")
    print("-" * 60)

    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            # 数据处理 - 兼容不同的数据加载器返回格式
            if len(data) == 2:
                # TestDataLoaderInference 格式: (data, filename)
                input_data = data[0].type(torch.FloatTensor).to(device)
                filename = data[1][0] if isinstance(data[1], (list, tuple)) else data[1]
                true_label = None  # 无真实标签
            elif len(data) == 3:
                # GesDataLoaderNew 格式: (data, label, filename)
                input_data = data[0].type(torch.FloatTensor).to(device)
                true_label = data[1].to(device) if data[1] is not None else None
                filename = data[2][0] if isinstance(data[2], (list, tuple)) else data[2]
            else:
                print(f"未知的数据格式: {len(data)} 个元素")
                continue

            print(f"样本 {step + 1:3d}: {filename}")

            # 模型推理
            outputs = net(input_data)

            # 获取6类的softmax概率
            probs_6class = torch.softmax(outputs, dim=1)

            print(f"  原始6类概率: {probs_6class.squeeze().cpu().numpy()}")

            # 只取前4类进行预测
            probs_4class = probs_6class[:, :4]

            # 重新归一化前4类的概率，使其和为1
            probs_4class_norm = probs_4class / probs_4class.sum(dim=1, keepdim=True)

            # 获取预测结果
            predicted_class = torch.argmax(probs_4class_norm, dim=1)
            confidence = torch.max(probs_4class_norm, dim=1)[0]

            print(f"  前4类概率: {probs_4class_norm.squeeze().cpu().numpy()}")
            print(f"  预测类别: {predicted_class.item()} ({target_classes[predicted_class.item()]})")
            print(f"  置信度: {confidence.item():.4f}")

            # 应用置信度阈值
            if confidence.item() >= conf_threshold:
                final_prediction = predicted_class.item()
                final_class_name = target_classes[final_prediction]
                status = "可信"
            else:
                final_prediction = -1  # 未识别/低置信度
                final_class_name = "未识别"
                status = "低置信度"

            print(f"  最终预测: {final_prediction} ({final_class_name}) - {status}")

            # 保存结果
            result = {
                'filename': filename,
                'prediction': final_prediction,
                'class_name': final_class_name,
                'confidence': confidence.item(),
                'status': status,
                'probabilities_4class': probs_4class_norm.squeeze().cpu().numpy().tolist(),
                'probabilities_6class': probs_6class.squeeze().cpu().numpy().tolist()
            }
            results.append(result)

            print()

    # 7. 结果统计
    print("=" * 60)
    print("推理结果统计")
    print("=" * 60)

    # 统计预测分布
    pred_counts = {}
    confidence_stats = {'high': 0, 'low': 0}

    for result in results:
        pred = result['prediction']
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

        if result['status'] == '可信':
            confidence_stats['high'] += 1
        else:
            confidence_stats['low'] += 1

    print("预测类别分布:")
    for pred, count in sorted(pred_counts.items()):
        if pred == -1:
            print(f"  未识别: {count}")
        else:
            print(f"  类别 {pred} ({target_classes[pred]}): {count}")

    print(f"\n置信度统计:")
    print(f"  高置信度 (>={conf_threshold}): {confidence_stats['high']}")
    print(f"  低置信度 (<{conf_threshold}): {confidence_stats['low']}")

    # 计算各类别的平均置信度
    class_confidences = {i: [] for i in range(4)}
    class_confidences[-1] = []  # 未识别类

    for result in results:
        if result['prediction'] != -1:
            class_confidences[result['prediction']].append(result['confidence'])
        else:
            class_confidences[-1].append(result['confidence'])

    print(f"\n各类别平均置信度:")
    for pred, confidences in class_confidences.items():
        if confidences:
            avg_conf = np.mean(confidences)
            if pred == -1:
                print(f"  未识别: {avg_conf:.4f}")
            else:
                print(f"  类别 {pred} ({target_classes[pred]}): {avg_conf:.4f}")

    # 如果有原始标签，计算准确率（仅供参考）
    # labeled_results = [r for r in results if r['original_label'] is not None and r['original_label'] < 4]
    # if labeled_results:
    #     correct_count = sum(1 for r in labeled_results if r['prediction'] == r['original_label'])
    #     accuracy = correct_count / len(labeled_results)
    #     print(f"\n参考准确率 (基于原始标签): {accuracy:.4f} ({correct_count}/{len(labeled_results)})")

    # 8. 保存结果
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)

    # 保存详细结果 (JSON)
    with open(output_dir / 'inference_results_4class_final.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存简化结果 (CSV)
    df_results = pd.DataFrame([
        {
            'filename': r['filename'],
            'prediction': r['prediction'],
            'class_name': r['class_name'],
            'confidence': r['confidence'],
            'status': r['status']
        } for r in results
    ])

    df_results.to_csv(output_dir / 'inference_results_4class_final.csv', index=False, encoding='utf-8')

    print(f"\n结果已保存到:")
    print(f"  详细结果: {output_dir / 'inference_results_4class_final.json'}")
    print(f"  简化结果: {output_dir / 'inference_results_4class_final.csv'}")

    # 9. 生成航迹级别的汇总
    generate_track_summary(results, output_dir, target_classes)


def generate_track_summary(results, output_dir, target_classes):
    """
    生成航迹级别的预测汇总
    """
    print(f"\n生成航迹级别汇总...")

    # 按航迹ID分组
    track_results = {}

    for result in results:
        filename = result['filename']

        # 从文件名提取航迹ID
        import re
        track_match = re.search(r'Track(\d+)_', filename)
        if track_match:
            track_id = int(track_match.group(1))

            if track_id not in track_results:
                track_results[track_id] = []

            track_results[track_id].append(result)

    # 为每个航迹生成汇总预测
    track_summary = {}

    for track_id, track_data in track_results.items():
        # 策略1：使用置信度最高的预测
        best_result = max(track_data, key=lambda x: x['confidence'])

        # 策略2：投票决定（只考虑高置信度的预测）
        high_conf_predictions = [r['prediction'] for r in track_data
                                 if r['confidence'] >= 0.5 and r['prediction'] != -1]

        if high_conf_predictions:
            from collections import Counter
            vote_result = Counter(high_conf_predictions).most_common(1)[0][0]
            vote_confidence = np.mean([r['confidence'] for r in track_data
                                       if r['prediction'] == vote_result])
        else:
            vote_result = -1
            vote_confidence = 0.0

        track_summary[track_id] = {
            'track_id': track_id,
            'n_matrices': len(track_data),
            'best_prediction': best_result['prediction'],
            'best_confidence': best_result['confidence'],
            'vote_prediction': vote_result,
            'vote_confidence': vote_confidence,
            'final_prediction': best_result['prediction'],  # 使用置信度最高的作为最终结果
            'final_class_name': best_result['class_name'],
            'matrix_files': [r['filename'] for r in track_data]
        }

    # 保存航迹汇总
    with open(output_dir / 'track_summary_4class_final.json', 'w', encoding='utf-8') as f:
        json.dump(track_summary, f, indent=2, ensure_ascii=False)

    # 生成提交格式
    submission_data = []
    for track_id, summary in sorted(track_summary.items()):
        submission_data.append({
            'track_id': track_id,
            'prediction': summary['final_prediction'],
            'class_name': summary['final_class_name'],
            'confidence': summary['best_confidence']
        })

    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv(output_dir / 'submission_4class_final.csv', index=False, encoding='utf-8')

    print(f"航迹汇总: {output_dir / 'track_summary_4class_final.json'}")
    print(f"提交文件: {output_dir / 'submission_4class_final.csv'}")

    # 显示航迹预测统计
    print(f"\n航迹级别预测统计:")
    track_pred_counts = {}
    for summary in track_summary.values():
        pred = summary['final_prediction']
        track_pred_counts[pred] = track_pred_counts.get(pred, 0) + 1

    for pred, count in sorted(track_pred_counts.items()):
        if pred == -1:
            print(f"  未识别: {count} 个航迹")
        else:
            print(f"  类别 {pred} ({target_classes[pred]}): {count} 个航迹")


if __name__ == "__main__":
    inference_main()