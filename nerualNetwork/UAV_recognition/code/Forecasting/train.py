import os.path
import time
import torch
import shutil
from torch import nn, optim
from torch.utils.data import DataLoader

from refinenet_4cascade import RefineNet4Cascade
from drsncww import rsnet34
from data_loader_new import GesDataLoaderNew

# from data_loader_3channels import GesDataLoaderNew
# from data_loader_2channels import GesDataLoaderNew

RESUME = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train_path1 = "./dataset/dataset_0717_abs_ALL/train"
train_path1 = "./dataset/TEST_0809/train"
train_path2 = "./dataset/train"
train_path3 = "./dataset/train"
val_path1 = "./dataset/TEST_0809/val"
val_path2 = "./dataset/val"
val_path3 = "./dataset/val"
s_path = './finalweights/todo/DRSN/refine_channel2_R_weight.pth'
# s_path = './weights/DRSNCW_channel1_V_weight.pth'
# 以上信息记得修改！！！

training_epoch = 100
train_dataset = GesDataLoaderNew(train_path1, data_rows=32, data_cols=64)  # 加载数据集
train_data_loader = DataLoader(train_dataset, batch_size=16,
                               num_workers=0, drop_last=True, shuffle=True)
val_dataset = GesDataLoaderNew(val_path1, data_rows=32, data_cols=64, val=True)
conf_thresh_for_val = 0.5

net = rsnet34()
# net = RefineNet4Cascade((3, 64), num_classes=6)
net.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=1e-4)  # 优化器（训练参数，学习率）

precision = 0.0
start_epoch = -1
if RESUME:
    if not os.path.exists("./checkpoint_16frames"):
        print("checkpoint文件夹不存在，将从头开始训练")
        RESUME = False
    else:
        path_checkpoint = "./checkpoint/ckpt_best_62_88.42.pth"
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net_weight'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        precision = checkpoint['precision']

for epoch in range(start_epoch + 1, training_epoch):
    net.train()  # 训练过程中开启 Dropout
    running_loss = 0.0  # 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()  # 对训练一个 epoch 计时
    n_batches = 0

    for step, data in enumerate(train_data_loader, start=0):  # 遍历训练集，step从0开始计算
        input_data = data[0].type(torch.FloatTensor).to(device)
        labels = data[1].to(device)
        optimizer.zero_grad()  # 清除历史梯度

        outputs = net(input_data.to(device))  # 正向传播
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        running_loss += loss.item()
        n_batches += 1

        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_data_loader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    avg_train_loss = running_loss/max(n_batches,1)
    print(f"\nEpoch {epoch} | train_loss={avg_train_loss:.4f}") #可以看每个批次的训练损失

    # 验证
    if epoch % 1 == 0:  # 每1个epoch评估一次
        outputs = outputs.cpu().tolist()[0]
        outputs = [round(i, 3) for i in outputs]
        print(" ")
        # print('current_epoch={} output={} loss={}'.format(epoch, outputs, round(loss.cpu().tolist(), 5)))
        print('current_epoch={} loss={}'.format(epoch, round(loss.cpu().tolist(), 5)))
        # 训练中评估
        print('评估')
        point = 0
        total_eval = 0
        with torch.no_grad():
            net.eval()
            # val_dataloader = DataLoader(val_dataset, batch_size=1,
            #                             num_workers=0, drop_last=True, shuffle=True)

            val_dataloader = DataLoader(val_dataset, batch_size=1,
                                        num_workers=0, drop_last=False, shuffle=False,pin_memory=True)
            #shuffle一般在训练集中打乱防止记住数据，测试的时候感觉没有必要，然后为了不让后面的分母变大，把droplast给关了，pin_memory可开可不开，主要是为了提升数据拷贝速度的。

            for step, data in enumerate(val_dataloader):
                input_data = data[0].type(torch.FloatTensor).to(device)
                labels = data[1].to(device)
                file_name_for_check = data[2]
                start_time = time.time()
                outputs = net(input_data.to(device))
                end_time = time.time()
                spend_time = end_time - start_time
                FPS = 1 / (spend_time + float('1e-8'))

                outputs_softmax = torch.softmax(outputs, dim=1)  # shape [batch_size, num_classes]
                max_softmax, pred = outputs_softmax.max(dim=1)  # max_softmax: 最大概率, pred: 预测类别

                # 拒识阈值判断
                if max_softmax.item() < conf_thresh_for_val:
                    continue  # 该样本被拒识，不计入准确率

                total_eval += 1  # 计入有效评估样本
                if pred.item() == labels.item():
                    point += 1

                # target_class = outputs.index(max(outputs))  # 返回最大值的索引
                #
                # labels = labels.cpu().numpy().tolist()[0]
                #
                # if target_class == labels:
                #     point = point + 1
            if total_eval>0:
                p = round(point / total_eval, 4) * 100
            else:
                p = 0.0
            print('准确率', p, "%")
            if p >= precision:
                torch.save(net.state_dict(), s_path)  # SAVE WEIGHT
                precision = p
                checkpoint = {
                    "net_weight": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'precision': precision
                }

                # shutil.rmtree('./checkpoint')

                if not os.path.isdir("./checkpoint/deltaVr01_500epoch"):
                    os.mkdir("./checkpoint/deltaVr01_500epoch")
                torch.save(checkpoint,
                           './checkpoint/deltaVr01_500epoch/ckpt_best_%s_%s.pth' % (str(epoch), str(precision)))

    # print()
    # print('%f s' % (time.perf_counter() - time_start))
    # torch.save(net, s_path)

if __name__ == '__main__':
    pass
