import torch
import torch.nn.functional as F
from binary_weight import BinaryWeight
from ternary import Ternary


class QLinear(torch.nn.Linear):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.input_quantizer = Ternary(QLinear.qa_config)
        self.weight_quantizer = BinaryWeight(QLinear.qw_config)

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        out = F.linear(input_t, weight_b, self.bias)
        return out
