import torch
import torch.nn as nn
import torch.nn.init as init
from qconv import QConv2d
from qlinear import QLinear
from utils import View

__all__ = ["alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(96, 256, 5, 1, 2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(256, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(384, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(384, 256, 3, 1, 1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            View(256 * 6 * 6),
            QLinear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            # nn.Dropout() if dropout else None,
            QLinear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout() if dropout else None,
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model


if __name__ == '__main__':
    m = alexnet()
    print(m)
    x = torch.randn(2, 3, 224, 224)
    print(m(x).shape)
