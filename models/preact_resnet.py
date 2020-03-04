import torch
import torch.nn as nn
import torch.nn.functional as F

from qconv import QConv2d

__all__ = ["PreActResNet", "preact_resnet_18", "preact_resnet_34", "preact_resnet_50"]


class ShortCutA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, y.shape[2:])
        x = self.relu(self.bn(x))
        if x.size(1) == y.size(1):
            return x + y
        elif x.size(1) > y.size(1):
            x[:, :y.size(1), ...] += y
            return x
        else:
            y[:, :x.size(1), :, :] += x
            return y


class ShortCutB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.conv(self.relu(self.bn(x))) + y


class ShortCutC(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, y.shape[2:])
        return self.conv(self.relu(self.bn(x))) + y


def ShortCut(x: torch.Tensor, y: torch.Tensor):
    return x + y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = QConv2d(in_planes, planes, 3, stride, 1)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = QConv2d(planes, planes, 3, 1, 1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = self.downsample(identity, x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)

        self.bn3 = nn.BatchNorm2d(planes)
        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.Identity()
        self.conv3 = QConv2d(planes, planes * 4, kernel_size=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = self.conv3(self.relu3(self.bn3(x)))
        x = self.downsample(identity, x)
        return x


class BasicBlockR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut):
        super(BasicBlockR, self).__init__()
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QConv2d(in_planes, planes, 3, stride, 1)

        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, 3, 1, 1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.bn1(self.relu1(x)))
        x = self.conv2(self.bn2(self.relu2(x)))
        x = self.downsample(identity, x)
        return x


class BottleneckR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleneckR, self).__init__()
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1)

        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)

        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.Identity()
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.conv3 = QConv2d(planes, planes * 4, kernel_size=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.bn1(self.relu1(x)))
        x = self.conv2(self.bn2(self.relu2(x)))
        x = self.conv3(self.bn3(self.relu3(x)))
        x = self.downsample(identity, x)
        return x


class BasicBlockBi(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = QConv2d(in_planes, planes, 3, stride, 1)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = QConv2d(planes, planes, 3, 1, 1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.downsample(x, self.conv1(self.relu1(self.bn1(x))))
        x = x + self.conv2(self.relu2(self.bn2(x)))
        return x


class PreActResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, shortcut=ShortCutC, small_stem=False):
        self.in_planes = 64
        super(PreActResNet, self).__init__()
        if small_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.Identity(),
                QConv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.Identity(),
                QConv2d(32, self.in_planes, 3, 1, 1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, shortcut, 64, layers[0])
        self.layer2 = self._make_layer(block, shortcut, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, shortcut, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, shortcut, 512, layers[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, shortcut, planes, blocks, stride=1):
        downsample = ShortCut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = shortcut(self.in_planes, planes * block.expansion, stride)
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


model_list = {
    "18": (BasicBlock, [2, 2, 2, 2]),
    "34": (BasicBlock, [3, 4, 6, 3]),
    "50": (Bottleneck, [3, 4, 6, 3]),
    "101": (Bottleneck, [3, 4, 23, 3]),
    "152": (Bottleneck, [3, 8, 36, 3]),
    "18r": (BasicBlockR, [2, 2, 2, 2]),
    "34r": (BasicBlockR, [3, 4, 6, 3]),
    "50r": (BottleneckR, [3, 4, 6, 3]),
    "101r": (BottleneckR, [3, 4, 23, 3]),
    "152r": (BottleneckR, [3, 8, 36, 3]),
    '18bi': (BasicBlockBi, [2, 2, 2, 2]),
    "34bi": (BasicBlockBi, [3, 4, 6, 3]),
}

shortcut_list = {
    'A': ShortCutA,
    'B': ShortCutB,
    'C': ShortCutC,
}


def preact_resnet(depth="18", shortcut='A', **kwargs):
    depth = str(depth)
    assert depth in model_list.keys(), "Only support depth={" + ",".join(map(str, model_list.keys())) + "}"
    assert shortcut in shortcut_list.keys(), "Only support shortcut={" + ",".join(shortcut_list.keys()) + "}"
    return PreActResNet(*model_list[depth], shortcut=shortcut_list[shortcut], **kwargs)


def preact_resnet_18(**kwargs):
    return PreActResNet(*model_list['18'], shortcut=shortcut_list['C'], **kwargs)


def preact_resnet_34(**kwargs):
    return PreActResNet(*model_list['34'], shortcut=shortcut_list['C'], **kwargs)


def preact_resnet_50(**kwargs):
    return PreActResNet(*model_list['50'], shortcut=shortcut_list['A'], **kwargs)


if __name__ == '__main__':
    import torch

    m_ = preact_resnet_18()
    print(m_)
    m_(torch.randn(2, 3, 224, 224)).sum().backward()
