import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary


class QConv2d(torch.nn.Conv2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.input_quantizer = Ternary(QConv2d.qa_config)
        self.weight_quantizer = BinaryWeight(QConv2d.qw_config)

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
