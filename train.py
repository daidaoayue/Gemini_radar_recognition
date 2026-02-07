import os.path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader


torch.backends.cudnn.benchmark = True 

from refinenet_4cascade import RefineNet4Cascade
from drsncww import rsnet34
from data_loader_new import GesDataLoaderNew

def main():
    # ================= 1. 核心配置区域 =================
    RESUME = True
    
    # 【显存保护】
    # 如果跑起来没问题，可以尝试改为 32。
    TRAIN_BATCH_SIZE = 16   
    
    # 【极速验证】验证集不存梯度，显存占用极低，可以设很大
    VAL_BATCH_SIZE = 64     
    
    # 【CPU加速】利用32核CPU，开4个进程预读取数据
    NUM_WORKERS = 4         
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 路径配置
    train_path1 = "./dataset/train/2026-1-14-rng43/train"
    val_path1 = "./dataset/train/2026-1-14-rng43/val"
    s_path = './finalweights/todo/mul_rec/best_RD.pth'

    # 自动创建权重保存目录 (防止报错)
    s_dir = os.path.dirname(s_path)
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)
        print(f"已创建权重目录: {s_dir}")

    # 自动创建绘图保存目录
    plot_save_path = './output/loss_plots'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    training_epoch = 200

    # ================= 2. 数据加载优化 =================
    print("正在加载数据集...")
    # pin_memory=True: 开启锁页内存，加快 CPU 到 GPU 的数据传输
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
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-4)

    # --- 新增：添加学习率调度器 ---
# mode='max': 因为我们要监控的是准确率(Accuracy)，准确率越大越好
# factor=0.5: 触发条件时，学习率减半 (new_lr = old_lr * 0.5)
# patience=5: 如果连续 5 个 Epoch 准确率都没有提升，就触发减速
# verbose=True: 打印出学习率调整的信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    precision = 0.0
    start_epoch = -1
    loss_history = []
    acc_history = []
    epochs_record = []

    # ================= 4. 断点续训加载 =================
    if RESUME:
        path_checkpoint = "./checkpoint/ckpt_best_62_88.42.pth"
        if os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net_weight'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            precision = checkpoint['precision']
            print(f"✅ 成功加载模型，从 Epoch {start_epoch} 继续训练，当前最佳准确率: {precision}%")
        else:
            print("⚠️ 未找到权重文件，将从头开始训练")

    # ================= 5. 训练主循环 =================
    print("开始训练...")
    for epoch in range(start_epoch + 1, training_epoch):
        net.train()
        running_loss = 0.0
        n_batches = 0
        
        # 记录每个 epoch 开始时间
        epoch_start_time = time.time()

        for step, data in enumerate(train_data_loader):
            # non_blocking=True 可以让数据传输和计算并行
            input_data = data[0].type(torch.FloatTensor).to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = net(input_data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1

            # 减少打印频率，避免刷屏
            if step % 10 == 0: 
                print(f"\rEpoch {epoch} | Step {step}/{len(train_data_loader)} | Loss: {loss.item():.4f}", end="")

        avg_train_loss = running_loss / max(n_batches, 1)
        time_elapsed = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} 完成 | 耗时: {time_elapsed:.2f}s | Avg Loss: {avg_train_loss:.4f}")

        loss_history.append(avg_train_loss)
        epochs_record.append(epoch)

        # ================= 6. 验证循环 (加速版) =================
        # 建议每 1 个 epoch 验证一次
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

                    # 批量处理拒识逻辑 (Batch Operation)
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
            scheduler.step(p)
            # 保存最佳模型
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

        # ================= 7. 绘图 (Loss & Accuracy) =================
        try:
            plt.figure(figsize=(12, 5))
            
            # Loss 曲线
            plt.subplot(1, 2, 1)
            plt.plot(epochs_record, loss_history, label='Train Loss', color='red')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

            # Accuracy 曲线
            plt.subplot(1, 2, 2)
            # 对齐长度（防止验证频率不一致导致绘图报错）
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
            plt.close() # 关闭画布释放内存
        except Exception as e:
            print(f"绘图跳过: {e}")

if __name__ == '__main__':
    # Windows下多进程必须以此方式启动
    main()