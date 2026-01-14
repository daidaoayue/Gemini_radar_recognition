import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from drsncww import rsnet34
# 导入修改后的数据加载器
from data_loader_new import GesDataLoaderNew, TestDataLoader
from plot_confusion import ConfusionMatrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    print("=" * 60)
    print("修复后的测试程序")
    print("=" * 60)

    # 1. 数据加载 - 两种方式选择一种
    test_path1 = './dataset/preprocess_data/test'

    # 方式1：使用修改后的原加载器（推荐）
    test_dataset = GesDataLoaderNew(test_path1, data_rows=32, data_cols=64, val=True, test_mode=True)

    # 方式2：使用专门的测试加载器
    # test_dataset = TestDataLoader(test_path1, data_rows=32, data_cols=64)

    # 2. 检查数据
    print(f"\n数据集大小: {len(test_dataset)}")

    # 检查前几个样本
    print("前3个样本:")
    for i in range(min(3, len(test_dataset))):
        data, label, filename = test_dataset[i]
        print(f"  样本 {i}: 标签={label}, 文件={filename}, 数据形状={data.shape}")

    # 检查标签分布
    all_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"标签分布: {dict(zip(unique_labels, counts))}")

    # 3. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 模型加载
    net = rsnet34()
    s_path = './checkpoint/ckpt_best_10_94.38.pth'

    # 修复torch.load警告
    checkpoint = torch.load(s_path, map_location=device, weights_only=False)

    if 'net_weight' in checkpoint:
        net.load_state_dict(checkpoint['net_weight'])
        print("✓ 使用 'net_weight' 加载模型")
    else:
        net.load_state_dict(checkpoint)
        print("✓ 直接加载模型")

    net.to(device)
    net.eval()

    # 5. 类别定义 - 确保有4个标签
    class_indict_4 = {
        "0": "1_Lightweight_UAV",
        "1": "2_Small_UAV",
        "2": "3Birds",
        "3": "4Fly_ball"
    }

    labels = ["1_Lightweight_UAV", "2_Small_UAV", "3Birds", "4Fly_ball"]
    print(f"类别标签: {labels}")

    # 6. 创建混淆矩阵
    confusion = ConfusionMatrix(num_classes=4, labels=labels)

    # 7. 测试循环
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0,
                                 drop_last=False, shuffle=False)

    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_true_labels = []

    print(f"\n开始测试...")
    print("-" * 40)

    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            input_data = data[0].type(torch.FloatTensor).to(device)
            true_label = data[1].to(device)
            filename = data[2][0] if isinstance(data[2], (list, tuple)) else data[2]

            print(f"样本 {step + 1}: {filename}, 真实标签={true_label.item()}")

            # 过滤掉不属于前4类的样本
            if true_label.item() >= 4:
                print(f"  跳过: 标签 >= 4")
                continue

            # 模型推理
            outputs = net(input_data)
            probs = torch.softmax(outputs, dim=1)

            print(f"  原始6类概率: {probs.squeeze().cpu().numpy()}")

            # 只考虑前4类，重新归一化
            probs_4class = probs[:, :4]
            probs_4class_norm = probs_4class / probs_4class.sum(dim=1, keepdim=True)

            predicted = torch.argmax(probs_4class_norm, dim=1)
            confidence = torch.max(probs_4class_norm, dim=1)[0]

            print(f"  前4类概率: {probs_4class_norm.squeeze().cpu().numpy()}")
            print(f"  预测: {predicted.item()}, 置信度: {confidence.item():.4f}")

            # 统计
            all_predictions.append(predicted.item())
            all_true_labels.append(true_label.item())

            correct = (predicted == true_label).item()
            correct_predictions += correct
            total_samples += 1

            print(f"  {'✓' if correct else '✗'} 预测{'正确' if correct else '错误'}")
            print()

    # 8. 结果统计
    print("=" * 60)
    print("测试结果")
    print("=" * 60)

    if total_samples > 0:
        # 手动更新混淆矩阵
        confusion.update(np.array(all_predictions), np.array(all_true_labels))

        overall_accuracy = correct_predictions / total_samples
        print(f"总样本数: {total_samples}")
        print(f"总体准确率: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")

        print(f"\n混淆矩阵:")
        print(confusion.matrix)

        # 分类别准确率
        print("\n分类别统计:")
        for i in range(4):
            class_mask = np.array(all_true_labels) == i
            if class_mask.any():
                class_predictions = np.array(all_predictions)[class_mask]
                class_accuracy = (class_predictions == i).sum() / len(class_predictions)
                print(
                    f"  {labels[i]:20s}: {class_accuracy:.4f} ({(class_predictions == i).sum()}/{len(class_predictions)})")

        # 9. 绘制结果
        try:
            confusion.plot()
            confusion.summary()
            print("✓ 混淆矩阵绘制成功")
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")

            # 使用matplotlib手动绘制
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 8))
                plt.imshow(confusion.matrix, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()

                tick_marks = np.arange(4)
                plt.xticks(tick_marks, labels, rotation=45, ha='right')
                plt.yticks(tick_marks, labels)

                # 添加数值
                thresh = confusion.matrix.max() / 2.
                for i in range(4):
                    for j in range(4):
                        plt.text(j, i, f'{confusion.matrix[i, j]:.0f}',
                                 horizontalalignment="center",
                                 color="white" if confusion.matrix[i, j] > thresh else "black")

                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.show()
                print("✓ 手动绘制混淆矩阵成功")

            except Exception as e2:
                print(f"手动绘制也失败: {e2}")

    else:
        print("没有有效的测试样本")


if __name__ == "__main__":
    main()