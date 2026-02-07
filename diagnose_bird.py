"""
鸟类识别诊断脚本
================
分析RD模型和Track模型分别对鸟类的识别能力，
找出瓶颈在哪里。

输出：
1. RD单独对鸟类的准确率
2. Track单独对鸟类的准确率
3. 融合后对鸟类的准确率
4. 鸟类被误判的模式分析
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


class TrackOnlyNetV3(nn.Module):
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


def main():
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    CHECKPOINT = "./checkpoint/fusion_v13_final/ckpt_best_98.43.pth"
    
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其它'}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("鸟类识别诊断")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    import glob
    if not os.path.exists(CHECKPOINT):
        pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
        if pths:
            CHECKPOINT = sorted(pths)[-1]
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', 0.40)
    rd_weight = 1.0 - track_weight
    
    # RD模型
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    # Track模型
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    print(f"融合权重: RD={rd_weight:.2f}, Track={track_weight:.2f}")
    
    # 收集预测结果
    print("\n分析各模型表现...")
    
    results = {
        'rd': [],      # RD单独预测
        'track': [],   # Track单独预测
        'fusion': [],  # 融合预测
    }
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            # RD预测
            rd_logits = rd_model(x_rd)
            rd_probs = torch.softmax(rd_logits, dim=1)
            rd_conf, rd_pred = rd_probs.max(dim=1)
            
            # Track预测
            track_logits = track_model(x_track, x_stats)
            track_probs = torch.softmax(track_logits, dim=1)
            track_conf, track_pred = track_probs.max(dim=1)
            
            # 融合预测
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            fused_conf, fused_pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                results['rd'].append({
                    'true': true_label,
                    'pred': rd_pred[i].item(),
                    'conf': rd_conf[i].item(),
                    'probs': rd_probs[i].cpu().numpy()
                })
                results['track'].append({
                    'true': true_label,
                    'pred': track_pred[i].item(),
                    'conf': track_conf[i].item(),
                    'probs': track_probs[i].cpu().numpy()
                })
                results['fusion'].append({
                    'true': true_label,
                    'pred': fused_pred[i].item(),
                    'conf': fused_conf[i].item(),
                    'probs': fused_probs[i].cpu().numpy()
                })
    
    # 分析各模型对各类别的表现
    print("\n" + "="*70)
    print("各模型对各类别的准确率（置信度≥0.5）")
    print("="*70)
    
    print(f"\n{'模型':^10}|{'轻型无人机':^12}|{'小型无人机':^12}|{'鸟类':^12}|{'空飘球':^12}|{'总体':^10}")
    print("-" * 75)
    
    for model_name in ['rd', 'track', 'fusion']:
        data = results[model_name]
        
        # 按类别统计
        class_acc = {}
        for c in VALID_CLASSES:
            c_data = [d for d in data if d['true'] == c and d['conf'] >= 0.5]
            c_correct = sum(1 for d in c_data if d['pred'] == d['true'])
            class_acc[c] = 100 * c_correct / len(c_data) if c_data else 0
        
        # 总体
        valid_data = [d for d in data if d['conf'] >= 0.5]
        total_correct = sum(1 for d in valid_data if d['pred'] == d['true'])
        total_acc = 100 * total_correct / len(valid_data) if valid_data else 0
        
        print(f"{model_name:^10}|{class_acc[0]:^12.1f}|{class_acc[1]:^12.1f}|{class_acc[2]:^12.1f}|{class_acc[3]:^12.1f}|{total_acc:^10.1f}")
    
    # 专门分析鸟类
    print("\n" + "="*70)
    print("鸟类样本详细分析")
    print("="*70)
    
    bird_data_rd = [d for d in results['rd'] if d['true'] == 2]
    bird_data_track = [d for d in results['track'] if d['true'] == 2]
    bird_data_fusion = [d for d in results['fusion'] if d['true'] == 2]
    
    print(f"\n鸟类样本总数: {len(bird_data_rd)}")
    
    # 各模型对鸟类的置信度分布
    print(f"\n鸟类样本置信度分布:")
    print(f"{'模型':^10}|{'<0.4':^8}|{'0.4-0.5':^8}|{'0.5-0.6':^8}|{'0.6-0.7':^8}|{'≥0.7':^8}|{'平均':^8}")
    print("-" * 65)
    
    for model_name, data in [('RD', bird_data_rd), ('Track', bird_data_track), ('Fusion', bird_data_fusion)]:
        bins = [0, 0, 0, 0, 0]
        for d in data:
            conf = d['conf']
            if conf < 0.4:
                bins[0] += 1
            elif conf < 0.5:
                bins[1] += 1
            elif conf < 0.6:
                bins[2] += 1
            elif conf < 0.7:
                bins[3] += 1
            else:
                bins[4] += 1
        
        avg_conf = np.mean([d['conf'] for d in data])
        print(f"{model_name:^10}|{bins[0]:^8}|{bins[1]:^8}|{bins[2]:^8}|{bins[3]:^8}|{bins[4]:^8}|{avg_conf:^8.3f}")
    
    # 鸟类被误判为什么
    print(f"\n鸟类被误判分析（融合模型，所有样本）:")
    bird_wrong = [d for d in bird_data_fusion if d['pred'] != 2]
    
    misclass_count = defaultdict(int)
    for d in bird_wrong:
        misclass_count[d['pred']] += 1
    
    print(f"  鸟类样本总数: {len(bird_data_fusion)}")
    print(f"  被误判数: {len(bird_wrong)}")
    for pred_class, count in sorted(misclass_count.items(), key=lambda x: -x[1]):
        print(f"    → {CLASS_NAMES.get(pred_class, f'类别{pred_class}')}: {count}个")
    
    # 其他类别被误判为鸟类
    print(f"\n其他类别被误判为鸟类（融合模型，所有样本）:")
    for true_class in [0, 1, 3]:
        class_data = [d for d in results['fusion'] if d['true'] == true_class]
        misclass_as_bird = [d for d in class_data if d['pred'] == 2]
        print(f"  {CLASS_NAMES[true_class]} → 鸟类: {len(misclass_as_bird)}个 / {len(class_data)}个")
    
    # 分析RD和Track对鸟类的"一致性"
    print("\n" + "="*70)
    print("RD与Track对鸟类预测的一致性分析")
    print("="*70)
    
    agree_correct = 0  # 两个模型都对
    agree_wrong = 0    # 两个模型都错
    rd_only_correct = 0  # 只有RD对
    track_only_correct = 0  # 只有Track对
    
    for i in range(len(bird_data_rd)):
        rd_correct = bird_data_rd[i]['pred'] == 2
        track_correct = bird_data_track[i]['pred'] == 2
        
        if rd_correct and track_correct:
            agree_correct += 1
        elif not rd_correct and not track_correct:
            agree_wrong += 1
        elif rd_correct:
            rd_only_correct += 1
        else:
            track_only_correct += 1
    
    print(f"\n鸟类样本预测一致性:")
    print(f"  RD和Track都正确: {agree_correct} ({100*agree_correct/len(bird_data_rd):.1f}%)")
    print(f"  RD和Track都错误: {agree_wrong} ({100*agree_wrong/len(bird_data_rd):.1f}%)")
    print(f"  只有RD正确: {rd_only_correct} ({100*rd_only_correct/len(bird_data_rd):.1f}%)")
    print(f"  只有Track正确: {track_only_correct} ({100*track_only_correct/len(bird_data_rd):.1f}%)")
    
    # 建议
    print("\n" + "="*70)
    print("诊断结论与建议")
    print("="*70)
    
    # 计算哪个模型对鸟类更好
    rd_bird_acc = sum(1 for d in bird_data_rd if d['pred'] == 2) / len(bird_data_rd)
    track_bird_acc = sum(1 for d in bird_data_track if d['pred'] == 2) / len(bird_data_track)
    fusion_bird_acc = sum(1 for d in bird_data_fusion if d['pred'] == 2) / len(bird_data_fusion)
    
    print(f"""
鸟类识别准确率（不带置信度过滤）:
  - RD模型: {100*rd_bird_acc:.1f}%
  - Track模型: {100*track_bird_acc:.1f}%
  - 融合模型: {100*fusion_bird_acc:.1f}%
""")
    
    if rd_bird_acc > track_bird_acc:
        print(f"结论: RD模型对鸟类识别更好 (+{100*(rd_bird_acc-track_bird_acc):.1f}%)")
        print(f"建议: 考虑增加RD权重，或改进Track模型的鸟类特征提取")
    else:
        print(f"结论: Track模型对鸟类识别更好 (+{100*(track_bird_acc-rd_bird_acc):.1f}%)")
        print(f"建议: 考虑增加Track权重，或改进RD模型")
    
    if agree_wrong > 20:
        print(f"\n警告: {agree_wrong}个鸟类样本两个模型都判断错误")
        print(f"这些可能是数据标注问题或真正的困难样本")
    
    # 计算最优融合权重（针对鸟类）
    print(f"\n" + "="*70)
    print("测试不同融合权重对鸟类的影响")
    print("="*70)
    
    print(f"\n{'Track权重':^12}|{'鸟类准确率':^12}|{'鸟类平均置信度':^15}|{'总体准确率':^12}")
    print("-" * 55)
    
    for tw in [0.2, 0.3, 0.4, 0.5, 0.6]:
        rw = 1.0 - tw
        
        # 重新计算融合结果
        bird_correct = 0
        bird_conf_sum = 0
        total_correct = 0
        total_count = 0
        
        for i in range(len(results['rd'])):
            rd_probs = results['rd'][i]['probs']
            track_probs = results['track'][i]['probs']
            true_label = results['rd'][i]['true']
            
            fused = rw * rd_probs + tw * track_probs
            pred = np.argmax(fused)
            conf = fused[pred]
            
            if conf >= 0.5:
                total_count += 1
                if pred == true_label:
                    total_correct += 1
            
            if true_label == 2:  # 鸟类
                bird_conf_sum += fused[2]  # 对鸟类的置信度
                if pred == 2:
                    bird_correct += 1
        
        bird_acc = 100 * bird_correct / len(bird_data_rd)
        bird_avg_conf = bird_conf_sum / len(bird_data_rd)
        total_acc = 100 * total_correct / total_count if total_count > 0 else 0
        
        mark = " *" if tw == track_weight else ""
        print(f"{tw:^12.1f}|{bird_acc:^12.1f}|{bird_avg_conf:^15.3f}|{total_acc:^12.2f}{mark}")


if __name__ == '__main__':
    main()