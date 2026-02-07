"""
双流融合模型测试脚本（最终版）
============================
功能：
1. 加载训练好的融合模型
2. 对测试数据进行预测
3. 输出每个样本的预测结果和置信度
4. 生成分类报告

使用方法：
    python test_fusion_final.py --rd_dir <RD数据目录> --track_dir <航迹特征目录>
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse
import warnings
import re

warnings.filterwarnings("ignore")

# 导入数据加载器和模型
from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


class TrackOnlyNetV3(nn.Module):
    """航迹特征网络"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten()
        )
        self.stats_net = nn.Sequential(
            nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


class FusionPredictor:
    """双流融合预测器"""
    
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 类别名称
        self.class_names = {
            0: '轻型旋翼无人机',
            1: '小型旋翼无人机',
            2: '鸟类',
            3: '空飘球',
            4: '杂波',
            5: '其它未识别目标'
        }
        
        # 加载checkpoint
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 获取融合权重（训练脚本保存的是 best_fixed_weight）
        self.track_weight = checkpoint.get('best_fixed_weight', checkpoint.get('track_weight', 0.40))
        self.rd_weight = 1.0 - self.track_weight
        print(f"融合权重: RD={self.rd_weight:.2f}, Track={self.track_weight:.2f}")
        
        # 加载RD模型
        self.rd_model = rsnet34()
        if 'rd_model' in checkpoint:
            self.rd_model.load_state_dict(checkpoint['rd_model'])
        else:
            # 训练脚本没有保存rd_model，需要从预训练路径加载
            import glob
            rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") + glob.glob("./checkpoint/*93*.pth")
            if rd_pths:
                rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
                rd_state = rd_ckpt.get('net_weight', rd_ckpt)
                self.rd_model.load_state_dict(rd_state)
                print(f"RD模型加载自: {rd_pths[0]}")
            else:
                print("警告: 未找到RD预训练模型")
        print("RD模型加载完成")
        
        # 加载航迹模型
        self.track_model = TrackOnlyNetV3()
        if 'track_model' in checkpoint:
            self.track_model.load_state_dict(checkpoint['track_model'])
        print("航迹模型加载完成")
        
        # 移动到设备
        self.rd_model.to(self.device).eval()
        self.track_model.to(self.device).eval()
    
    def predict_batch(self, x_rd, x_track, x_stats):
        """批量预测"""
        with torch.no_grad():
            x_rd = x_rd.to(self.device)
            x_track = x_track.to(self.device)
            x_stats = x_stats.to(self.device)
            
            # 获取两个模型的输出
            rd_logits = self.rd_model(x_rd)
            track_logits = self.track_model(x_track, x_stats)
            
            # 概率融合
            rd_probs = torch.softmax(rd_logits, dim=1)
            track_probs = torch.softmax(track_logits, dim=1)
            fused_probs = self.rd_weight * rd_probs + self.track_weight * track_probs
            
            # 获取预测结果
            confidences, predictions = fused_probs.max(dim=1)
        
        return predictions.cpu().numpy(), confidences.cpu().numpy(), fused_probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='双流融合模型测试')
    parser.add_argument('--rd_dir', type=str, default='./dataset/train_cleandata/val',
                        help='RD数据目录')
    parser.add_argument('--track_dir', type=str, default='./dataset/track_enhanced_cleandata/val',
                        help='航迹特征目录')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/fusion_v13_final/ckpt_best_98.43.pth',
                        help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--output', type=str, default='predictions.txt', help='输出文件')
    parser.add_argument('--valid_classes', type=str, default='0,1,2,3', help='有效类别（比赛只识别前4类）')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='置信度阈值（与训练一致）')
    
    args = parser.parse_args()
    
    # 解析有效类别
    valid_classes = [int(c) for c in args.valid_classes.split(',')]
    conf_thresh = args.conf_thresh
    
    print(f"\n{'='*60}")
    print(f"双流融合模型测试")
    print(f"{'='*60}")
    print(f"RD数据目录: {args.rd_dir}")
    print(f"航迹特征目录: {args.track_dir}")
    print(f"模型检查点: {args.checkpoint}")
    print(f"有效类别: {valid_classes}")
    print(f"置信度阈值: {conf_thresh}")
    
    # 检查路径
    import glob
    if not os.path.exists(args.checkpoint):
        pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
        if pths:
            args.checkpoint = sorted(pths)[-1]
            print(f"使用找到的checkpoint: {args.checkpoint}")
        else:
            print(f"错误: 找不到模型文件")
            return
    
    # 加载数据
    print(f"\n加载测试数据...")
    test_ds = FusionDataLoaderV3(args.rd_dir, args.track_dir, val=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"测试样本数: {len(test_ds)}")
    
    # 创建预测器
    predictor = FusionPredictor(args.checkpoint)
    
    # 预测
    print(f"\n开始预测...")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_filenames = []
    
    sample_idx = 0
    for x_rd, x_track, x_stats, y in test_loader:
        preds, confs, probs = predictor.predict_batch(x_rd, x_track, x_stats)
        
        for i in range(len(preds)):
            if sample_idx < len(test_ds.samples):
                label, rd_path, _ = test_ds.samples[sample_idx]
                filename = os.path.basename(rd_path)
            else:
                filename = f'sample_{sample_idx}'
            
            all_predictions.append(preds[i])
            all_labels.append(y[i].item())
            all_confidences.append(confs[i])
            all_filenames.append(filename)
            
            sample_idx += 1
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # 计算准确率（过滤有效类别 + 置信度阈值）
    valid_mask = np.isin(all_labels, valid_classes) & (all_confidences >= conf_thresh)
    valid_preds = all_predictions[valid_mask]
    valid_labels = all_labels[valid_mask]
    valid_confs = all_confidences[valid_mask]
    valid_filenames = np.array(all_filenames)[valid_mask]
    
    # 统计被过滤掉的样本
    total_class_valid = np.sum(np.isin(all_labels, valid_classes))
    filtered_by_conf = total_class_valid - len(valid_labels)
    
    accuracy = 100 * np.mean(valid_preds == valid_labels)
    
    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"总样本数: {len(all_predictions)}")
    print(f"类别0-3样本数: {total_class_valid}")
    print(f"置信度过滤掉: {filtered_by_conf}个 (conf < {conf_thresh})")
    print(f"有效样本数: {len(valid_labels)}")
    print(f"覆盖率: {100 * len(valid_labels) / total_class_valid:.1f}%")
    print(f"准确率: {accuracy:.2f}%")
    
    # 分类别统计
    print(f"\n{'类别':^20}|{'样本数':^8}|{'正确数':^8}|{'准确率':^10}|{'平均置信度':^12}")
    print("-" * 65)
    
    for c in valid_classes:
        mask = valid_labels == c
        c_preds = valid_preds[mask]
        c_labels = valid_labels[mask]
        c_confs = valid_confs[mask]
        
        c_correct = np.sum(c_preds == c_labels)
        c_acc = 100 * c_correct / len(c_labels) if len(c_labels) > 0 else 0
        c_conf = np.mean(c_confs) if len(c_confs) > 0 else 0
        
        class_name = predictor.class_names.get(c, f'类别{c}')
        print(f"{class_name:^20}|{len(c_labels):^8}|{c_correct:^8}|{c_acc:^10.2f}%|{c_conf:^12.3f}")
    
    # 混淆矩阵
    print(f"\n混淆矩阵（行=真实，列=预测）:")
    print(f"{'':^8}", end='')
    for c in valid_classes:
        print(f"{c:^8}", end='')
    print()
    
    for true_c in valid_classes:
        print(f"{true_c:^8}", end='')
        for pred_c in valid_classes:
            count = np.sum((valid_labels == true_c) & (valid_preds == pred_c))
            print(f"{count:^8}", end='')
        print()
    
    # 保存预测结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("文件名\t真实标签\t预测标签\t置信度\t是否正确\n")
        for i in range(len(valid_labels)):
            correct = '✓' if valid_labels[i] == valid_preds[i] else '✗'
            f.write(f"{valid_filenames[i]}\t{valid_labels[i]}\t{valid_preds[i]}\t{valid_confs[i]:.4f}\t{correct}\n")
    
    print(f"\n预测结果已保存到: {args.output}")
    
    # 错误分析
    wrong_indices = np.where(valid_preds != valid_labels)[0]
    
    print(f"\n{'='*60}")
    print(f"错误分析（共{len(wrong_indices)}个错误）")
    print(f"{'='*60}")
    
    if len(wrong_indices) > 0 and len(wrong_indices) <= 30:
        print(f"\n{'序号':^6}|{'真实':^12}|{'预测':^12}|{'置信度':^8}|{'文件名':^30}")
        print("-" * 75)
        
        for idx in wrong_indices[:30]:
            true_name = predictor.class_names.get(valid_labels[idx], f'{valid_labels[idx]}')[:10]
            pred_name = predictor.class_names.get(valid_preds[idx], f'{valid_preds[idx]}')[:10]
            fname = valid_filenames[idx][:28] if len(valid_filenames[idx]) > 28 else valid_filenames[idx]
            print(f"{idx:^6}|{true_name:^12}|{pred_name:^12}|{valid_confs[idx]:^8.3f}|{fname:^30}")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    
    # 计算不带置信度过滤的准确率（用于对比）
    all_valid_mask = np.isin(all_labels, valid_classes)
    all_valid_acc = 100 * np.mean(all_predictions[all_valid_mask] == all_labels[all_valid_mask])
    
    print(f"  全部样本准确率（无置信度过滤）: {all_valid_acc:.2f}%")
    print(f"✓ 高置信度样本准确率: {accuracy:.2f}%")
    print(f"  覆盖率: {100 * len(valid_labels) / total_class_valid:.1f}%")
    
    if accuracy >= 98:
        print(f"\n✓ 已达到98%目标！")
    else:
        print(f"\n✗ 距离98%还差: {98 - accuracy:.2f}%")


if __name__ == '__main__':
    main()