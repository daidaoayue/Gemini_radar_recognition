"""
分析微多普勒特征的区分性
检查鸟类和无人机的特征是否有显著差异
"""

import os
import scipy.io as scio
import numpy as np
import re
from collections import defaultdict

# 路径
MD_DIR = "./dataset/micro_doppler_raw"

# 特征名称
feature_names = [
    'stft_bandwidth',        # 0
    'stft_centroid_std',     # 1
    'periodicity_2_15hz',    # 2 ← 翅膀拍动 (关键!)
    'periodicity_50_200hz',  # 3 ← 螺旋桨
    'modulation_depth',      # 4
    'spectral_entropy',      # 5
    'time_freq_correlation', # 6
    'micro_doppler_energy',  # 7
    'peak_periodicity_freq', # 8
    'sidelobe_level',        # 9
    'doppler_spread',        # 10
    'temporal_variance',     # 11
]

class_names = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}

def load_all_features():
    """加载所有微多普勒特征"""
    features_by_class = defaultdict(list)
    
    if not os.path.exists(MD_DIR):
        print(f"目录不存在: {MD_DIR}")
        return features_by_class
    
    for f in os.listdir(MD_DIR):
        if not f.endswith('.mat'):
            continue
        
        # 解析文件名: Track1_Label0_microdoppler.mat
        match = re.search(r'Track(\d+)_Label(\d+)', f)
        if not match:
            continue
        
        label = int(match.group(2))
        
        try:
            mat = scio.loadmat(os.path.join(MD_DIR, f))
            if 'micro_doppler_features' in mat:
                feat = mat['micro_doppler_features'].flatten()
                features_by_class[label].append(feat)
        except:
            pass
    
    return features_by_class

def analyze_features():
    print("=" * 70)
    print("微多普勒特征区分性分析")
    print("=" * 70)
    
    features_by_class = load_all_features()
    
    if not features_by_class:
        print("未找到特征文件")
        return
    
    # 统计
    print("\n样本数量:")
    for label in sorted(features_by_class.keys()):
        if label <= 3:
            print(f"  {class_names[label]}: {len(features_by_class[label])}")
    
    # 转换为numpy数组
    class_features = {}
    for label in [0, 1, 2, 3]:
        if label in features_by_class and len(features_by_class[label]) > 0:
            class_features[label] = np.array(features_by_class[label])
    
    if 2 not in class_features or 0 not in class_features:
        print("缺少鸟类或无人机数据")
        return
    
    bird_feat = class_features[2]
    uav_feat = class_features[0]
    
    print("\n" + "=" * 70)
    print("鸟类 vs 轻型无人机 特征对比")
    print("=" * 70)
    print(f"\n{'特征名':<25} | {'鸟类均值':>10} | {'无人机均值':>10} | {'差异%':>10} | 显著性")
    print("-" * 75)
    
    significant_features = []
    
    for i, name in enumerate(feature_names):
        bird_mean = np.mean(bird_feat[:, i])
        uav_mean = np.mean(uav_feat[:, i])
        bird_std = np.std(bird_feat[:, i])
        uav_std = np.std(uav_feat[:, i])
        
        if abs(uav_mean) > 0.001:
            diff_pct = 100 * (bird_mean - uav_mean) / abs(uav_mean)
        else:
            diff_pct = 0
        
        # 简单t检验
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(bird_feat[:, i], uav_feat[:, i])
        except:
            p_value = 1.0
        
        if p_value < 0.01 and abs(diff_pct) > 20:
            sig = "*** 极显著"
            significant_features.append(name)
        elif p_value < 0.05 and abs(diff_pct) > 10:
            sig = "** 显著"
            significant_features.append(name)
        elif p_value < 0.1:
            sig = "* 轻微"
        else:
            sig = ""
        
        # 标记关键特征
        key_mark = ""
        if i == 2:
            key_mark = " ← 翅膀拍动"
        elif i == 3:
            key_mark = " ← 螺旋桨"
        
        print(f"{name:<25} | {bird_mean:>10.4f} | {uav_mean:>10.4f} | {diff_pct:>+10.1f}% | {sig}{key_mark}")
    
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)
    
    # 检查关键特征
    bird_wing = np.mean(bird_feat[:, 2])  # periodicity_2_15hz
    uav_wing = np.mean(uav_feat[:, 2])
    bird_prop = np.mean(bird_feat[:, 3])  # periodicity_50_200hz
    uav_prop = np.mean(uav_feat[:, 3])
    
    print(f"\n翅膀拍动特征 (periodicity_2_15hz):")
    print(f"  鸟类: {bird_wing:.4f}")
    print(f"  无人机: {uav_wing:.4f}")
    print(f"  差异: {100*(bird_wing-uav_wing)/max(uav_wing, 0.001):+.1f}%")
    
    if abs(bird_wing - uav_wing) < 0.5:
        print("  ⚠️ 警告: 鸟类和无人机的翅膀拍动特征几乎没有差异！")
        print("  可能原因: 数据时长不足以检测2-15Hz的周期性")
    
    print(f"\n螺旋桨特征 (periodicity_50_200hz):")
    print(f"  鸟类: {bird_prop:.4f}")
    print(f"  无人机: {uav_prop:.4f}")
    print(f"  差异: {100*(bird_prop-uav_prop)/max(uav_prop, 0.001):+.1f}%")
    
    print(f"\n显著差异特征: {significant_features if significant_features else '无'}")
    
    if not significant_features:
        print("\n⚠️ 结论: 微多普勒特征在鸟类和无人机之间没有显著差异！")
        print("   这解释了为什么V17效果反而变差 - 这些特征是噪声而非有用信号。")
    else:
        print(f"\n✓ 有 {len(significant_features)} 个显著差异特征")
    
    # 检查数据时长问题
    print("\n" + "=" * 70)
    print("数据时长分析")
    print("=" * 70)
    print("""
    翅膀拍动频率: 2-15 Hz → 周期: 67-500 ms
    
    如果一个Track只有约100帧，每帧约0.85ms:
    总时长 ≈ 85ms
    
    这意味着数据只能捕捉到约 0.17-1.3 个翅膀拍动周期
    → 周期性检测的结果不可靠！
    
    建议: 
    1. 放弃微多普勒方案（数据不支持）
    2. 或者使用更长的观测窗口（如果有的话）
    """)


if __name__ == '__main__':
    analyze_features()