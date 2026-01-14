import torch.nn as nn
import torchvision.models as models
from resnet import resnet34
from blocks import (RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling)


class BaseRefineNet4Cascade(nn.Module):
    def __init__(self,
                 input_shape,
                 refinenet_block,
                 num_classes=6,
                 features=256,
                 resnet_factory=resnet34(),
                 # resnet_factory=models.resnet34,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int, int)): (channel, height, width)
                For 32x64 input: (0, 32, 64)
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()

        input_channel, input_height, input_width = input_shape

        # if input_size % 32 != 0:
        #     raise ValueError("{} not divisble by 32".format(input_shape))
        if input_height % 32 != 0 or input_width % 32 != 0:
            raise ValueError("Input dimensions {} must be divisible by 32".format(
                (input_height, input_width)))

        # resnet = resnet_factory(pretrained=pretrained)
        resnet = resnet_factory
        # self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
        #                             resnet.avgpool, resnet.layer1)
        if input_channel == 1:
            # 替换第一个卷积层
            original_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if input_channel == 2:
            # 替换第一个卷积层
            original_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)


        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.layer1_rn = nn.Conv2d(
            64, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            128, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            512, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        # self.refinenet4 = RefineNetBlock(2 * features,
        #                                  (2 * features, input_size // 32))
        # self.refinenet3 = RefineNetBlock(features,
        #                                  (2 * features, input_size // 32),
        #                                  (features, input_size // 16))
        # self.refinenet2 = RefineNetBlock(features,
        #                                  (features, input_size // 16),
        #                                  (features, input_size // 8))
        # self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
        #                                  (features, input_size // 4))
        h4, w4 = input_height // 32, input_width // 32  # 1x2
        h3, w3 = input_height // 16, input_width // 16  # 2x4
        h2, w2 = input_height // 8, input_width // 8  # 4x8
        h1, w1 = input_height // 4, input_width // 4  # 8x16
        #
        # RefineNet blocks
        self.refinenet4 = refinenet_block(2 * features, (2 * features, h4, w4))
        self.refinenet3 = refinenet_block(features,
                                          (2 * features, h4, w4),
                                          (features, h3, w3))
        self.refinenet2 = refinenet_block(features,
                                          (features, h3, w3),
                                          (features, h2, w2))
        self.refinenet1 = refinenet_block(features,
                                          (features, h2, w2),
                                          (features, h1, w1))



        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        out = self.output_conv(path_1)
        out1 = self.avgpool(out)
        out2 = out1.view(out1.size(0), -1)
        out2 = self.fc(out2)
        return out2


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=6,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


class RefineNet4Cascade(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=6,
                 features=256,
                 resnet_factory=resnet34(),
                 # resnet_factory=models.resnet34,
                 pretrained=False,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int, int)): (channel, height, width)
                For 32x64 input: (0, 32, 64)
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlock,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)
