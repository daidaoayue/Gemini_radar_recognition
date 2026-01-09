import os
import json
import torch
from torch.utils.data import DataLoader
from drsncww import rsnet34
# from refinenet_4cascade import RefineNet4Cascade
from data_loader_new import GesDataLoaderNew
# from data_loader_2channels import GesDataLoaderNew
# from data_loader_3channels import GesDataLoaderNew
from plot_confusion import ConfusionMatrix

test_path1 = "./dataset/data0507-1/test"
test_path2 = "./dataset/data0507-2/test"
test_path3 = "./dataset/data0507-4/test"
test_dataset = GesDataLoaderNew(test_path1, data_rows=64, data_cols=64, val=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = rsnet34()
# net = RefineNet4Cascade((3, 64), num_classes=9)
# s_path = './finalweights/refine_channel3_weight.pth'
s_path = './finalweights/DRSN/DRSNCW_channel1_V_weight.pth'
checkpoint = torch.load(s_path, map_location=torch.device('cpu'))
# 加载权重
net.load_state_dict(checkpoint, False)

conf_thresh = 0.5

# read class_indict
json_label_path = './class_indices.json'
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
class_indict = json.load(json_file)

labels = [label for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=9, labels=labels)

with torch.no_grad():
    for epoch in range(1):  #
        net.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     num_workers=0, drop_last=True, shuffle=True)

        for step, data in enumerate(test_dataloader):
            input_data = data[0].type(torch.FloatTensor).to(device)
            label = data[1].to(device)

            outputs = net(input_data.to(device))

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), label.to("cpu").numpy())

        confusion.plot()
        # confusion.summary()
