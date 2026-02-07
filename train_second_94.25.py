import os.path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

# 开启 cudnn 自动寻优
torch.backends.cudnn.benchmark = True 

from refinenet_4cascade import RefineNet4Cascade
from drsncww import rsnet34
from data_loader_new import GesDataLoaderNew

# ================= [新增] Mixup 数据增强函数 =================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''返回混合后的输入、成对的标签和混合比例 lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''混合后的 Loss 计算公式'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# ==========================================================

def main():
    # ================= 1. 核心配置区域 =================
    RESUME = True
    
    # 【显存保护】实际显存中每次只放 16 张图
    TRAIN_BATCH_SIZE = 16   
    
    # --- 梯度累积设置 ---
    # 目标等效 Batch Size = 64
    # 累积步数 = 64 / 16 = 4
    ACCUMULATION_STEPS = 4
    # -----------------------------
    
    # 【极速验证】
    VAL_BATCH_SIZE = 64     
    
    # 【CPU加速】
    NUM_WORKERS = 4         
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    print(f"策略: 物理Batch={TRAIN_BATCH_SIZE}, 累积步数={ACCUMULATION_STEPS}, 等效Batch={TRAIN_BATCH_SIZE * ACCUMULATION_STEPS}")

    # ================= [恢复你的原始路径] =================
    # 这里完全保留你之前的写法，不乱改
    # train_path1 = "./dataset/train/2026-1-14/train"
    # val_path1 = "./dataset/train/2026-1-14/val"
    train_path1 = "./dataset/train_cleandata/train"
    val_path1 = "./dataset/train_cleandata/val"
    s_path = './finalweights/todo/mul_rec/best_RD.pth'
    # 自动创建目录
    s_dir = os.path.dirname(s_path)
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)
        print(f"已创建权重目录: {s_dir}")

    plot_save_path = './output/loss_plots'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    training_epoch = 200

    # ================= 2. 数据加载优化 =================
    print("正在加载数据集...")
    train_dataset = GesDataLoaderNew(train_path1, data_rows=32, data_cols=64)
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS, 
        drop_last=True, 
        shuffle=True,
        pin_memory=True 
    )

    val_dataset = GesDataLoaderNew(val_path1, data_rows=32, data_cols=64, val=True)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=VAL_BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        drop_last=False, 
        shuffle=False,
        pin_memory=True
    )
    print(f"加载完成: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张")

    conf_thresh_for_val = 0.5

    # ================= 3. 模型与优化器 =================
    net = rsnet34()
    net.to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-3)

    # [修改] 调度器配置：改用余弦退火 (CosineAnnealingLR)
    # T_max 设置为总 epoch 数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epoch, eta_min=1e-6)
    print("已启用 CosineAnnealingLR 学习率调度器")

    precision = 0.0
    start_epoch = -1
    loss_history = []
    acc_history = []
    epochs_record = []

    # ================= 4. 断点续训加载 =================
    if RESUME:
        path_checkpoint = "./checkpoint/ckpt_best_last.pth" 
        # 注意：你需要确保这个文件存在，或者手动修改路径，否则会从头开始
        
        if os.path.exists(path_checkpoint):
            try:
                checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
                net.load_state_dict(checkpoint['net_weight'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                precision = checkpoint['precision']
                # scheduler.load_state_dict(checkpoint['scheduler']) 
                print(f"✅ 成功加载模型，从 Epoch {start_epoch} 继续训练，当前最佳准确率: {precision}%")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}，将从头开始训练")
        else:
            print("⚠️ 未找到权重文件，将从头开始训练")

    # ================= 5. 训练主循环 =================
    print("开始训练...")
    
    optimizer.zero_grad()
    
    for epoch in range(start_epoch + 1, training_epoch):
        net.train()
        running_loss = 0.0
        n_batches = 0
        
        epoch_start_time = time.time()

        for step, data in enumerate(train_data_loader):
            input_data = data[0].type(torch.FloatTensor).to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            
            # [新增] Mixup 训练逻辑
            # 1. 生成混合数据
            inputs, targets_a, targets_b, lam = mixup_data(input_data, labels, alpha=1.0)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            
            # 2. 前向传播 (使用混合后的 inputs)
            outputs = net(inputs)
            
            # 3. 计算混合 Loss
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
            
            # --- 梯度累积 ---
            # Loss 缩放
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()
            
            # 梯度更新
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计 Loss (乘回去以便显示)
            running_loss += loss.item() * ACCUMULATION_STEPS
            n_batches += 1

            if step % 20 == 0: 
                print(f"\rEpoch {epoch} | Step {step}/{len(train_data_loader)} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f}", end="")

        avg_train_loss = running_loss / max(n_batches, 1)
        time_elapsed = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} 完成 | 耗时: {time_elapsed:.2f}s | Avg Loss: {avg_train_loss:.4f}")

        loss_history.append(avg_train_loss)
        epochs_record.append(epoch)

        # ================= 6. 验证循环 =================
        if epoch % 1 == 0:
            print('正在评估...', end='')
            point = 0
            total_eval = 0
            
            with torch.no_grad():
                net.eval()
                for data in val_dataloader:
                    input_data = data[0].type(torch.FloatTensor).to(device, non_blocking=True)
                    labels = data[1].to(device, non_blocking=True)
                    
                    outputs = net(input_data)
                    outputs_softmax = torch.softmax(outputs, dim=1)
                    max_softmax, pred = outputs_softmax.max(dim=1)

                    valid_mask = max_softmax >= conf_thresh_for_val
                    if valid_mask.sum() > 0:
                        valid_pred = pred[valid_mask]
                        valid_labels = labels[valid_mask]
                        point += (valid_pred == valid_labels).sum().item()
                        total_eval += valid_mask.sum().item()

            if total_eval > 0:
                p = round(point / total_eval, 4) * 100
            else:
                p = 0.0
            
            acc_history.append(p)
            print(f'\r评估完成 | 准确率: {p:.2f}% | 历史最佳: {precision:.2f}%')
            
            # [修改] 学习率调度器更新位置：移到这里，不再传入 p
            scheduler.step()
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

            if p >= precision:
                torch.save(net.state_dict(), s_path)
                precision = p
                checkpoint = {
                    "net_weight": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'precision': precision
                }
                ckpt_dir = "./checkpoint/deltaVr01_200epoch"
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(checkpoint, f'{ckpt_dir}/ckpt_best_{epoch}_{precision}.pth')

        # ================= 7. 绘图 =================
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs_record, loss_history, label='Train Loss', color='red')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            if len(acc_history) == len(epochs_record):
                plt.plot(epochs_record, acc_history, label='Val Accuracy', color='blue')
            else:
                plt.plot(acc_history, label='Val Accuracy', color='blue')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plot_save_path, 'training_curves.png'))
            plt.close()
        except Exception as e:
            print(f"绘图跳过: {e}")

if __name__ == '__main__':
    main()