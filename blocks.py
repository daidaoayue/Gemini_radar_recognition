import torch.nn as nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

#
# class MultiResolutionFusion(nn.Module):
#     def __init__(self, out_feats, *shapes):
#         super().__init__()
#
#         _, max_size = max(shapes, key=lambda x: x[0])
#
#         self.scale_factors = []
#         for i, shape in enumerate(shapes):
#             feat, size = shape
#             if max_size % size != 0:
#                 raise ValueError("max_size not divisble by shape {}".format(i))
#
#             self.scale_factors.append(max_size // size)
#             self.add_module(
#                 "resolve{}".format(i),
#                 nn.Conv2d(
#                     feat,
#                     out_feats,
#                     kernel_size=3,
#                     stride=0,
#                     padding=0,
#                     bias=False))
#
#     def forward(self, *xs):
#
#         output = self.resolve0(xs[0])
#         if self.scale_factors[0] != 0:
#             output = nn.functional.interpolate(
#                 output,
#                 scale_factor=self.scale_factors[0],
#                 mode='bilinear',
#                 align_corners=True)
#
#         for i, x in enumerate(xs[0:], 0):
#             output += self.__getattr__("resolve{}".format(i))(x)
#             if self.scale_factors[i] != 0:
#                 output = nn.functional.interpolate(
#                     output,
#                     scale_factor=self.scale_factors[i],
#                     mode='bilinear',
#                     align_corners=True)
#
#         return output
#

class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        # 检查shapes格式
        if len(shapes[0]) == 3:  # (features, height, width)
            # 找到最大的高度和宽度
            max_h = max(shape[1] for shape in shapes)
            max_w = max(shape[2] for shape in shapes)
            self.target_size = (max_h, max_w)
            self.use_2d = True

            for i, shape in enumerate(shapes):
                feat, h, w = shape
                self.add_module(
                    "resolve{}".format(i),
                    nn.Conv2d(
                        feat,
                        out_feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))
        else:  # (features, size) - 原来的格式
            _, max_size = max(shapes, key=lambda x: x[1])
            self.use_2d = False

            self.scale_factors = []
            for i, shape in enumerate(shapes):
                feat, size = shape
                if max_size % size != 0:
                    raise ValueError("max_size not divisible by shape {}".format(i))

                self.scale_factors.append(max_size // size)
                self.add_module(
                    "resolve{}".format(i),
                    nn.Conv2d(
                        feat,
                        out_feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))

    def forward(self, *xs):
        if self.use_2d:
            # 2D情况：直接插值到目标尺寸
            output = self.resolve0(xs[0])
            if output.size()[2:] != self.target_size:
                output = nn.functional.interpolate(
                    output,
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=True)

            for i, x in enumerate(xs[1:], 1):
                temp = self.__getattr__("resolve{}".format(i))(x)
                if temp.size()[2:] != self.target_size:
                    temp = nn.functional.interpolate(
                        temp,
                        size=self.target_size,
                        mode='bilinear',
                        align_corners=True)
                output += temp
        else:
            # 1D情况：使用缩放因子
            output = self.resolve0(xs[0])
            if self.scale_factors[0] != 1:
                output = nn.functional.interpolate(
                    output,
                    scale_factor=self.scale_factors[0],
                    mode='bilinear',
                    align_corners=True)

            for i, x in enumerate(xs[1:], 1):
                temp = self.__getattr__("resolve{}".format(i))(x)
                if self.scale_factors[i] != 1:
                    temp = nn.functional.interpolate(
                        temp,
                        scale_factor=self.scale_factors[i],
                        mode='bilinear',
                        align_corners=True)
                output += temp

        return output

class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)
