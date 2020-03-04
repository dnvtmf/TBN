import torch
import torch.nn as nn

__all__ = ['BinaryWeight']


class binary_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_f):
        scale = weight_f.abs().sum(dim=list(range(1, weight_f.ndim)), keepdim=True) / weight_f[0].numel()
        weight_b = weight_f.sign() * scale
        return weight_b

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


class BinaryWeight(nn.Module):
    def __init__(self, config: dict, *args, **kwargs):
        super().__init__()
        self.config = config

    def forward(self, weight_f):
        weight_b = binary_weight.apply(weight_f)
        return weight_b
