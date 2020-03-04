import torch

__all__ = ['Ternary', 'ternary']


def _ternary(x: torch.Tensor, delta: float):
    return (x >= delta).float() - (x <= -delta).float()


class _ternary_without_scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs) -> torch.Tensor:
        input_f, running_delta, delta, momentum, training = inputs
        if momentum > 0:
            if training:
                ctx.delta = input_f.norm(1).item() * (delta / input_f.numel())  # = delta * |input_f|_1 / n
                running_delta.data = momentum * ctx.delta + (1.0 - momentum) * running_delta.data
            else:
                ctx.delta = running_delta.data.item()
        else:
            ctx.delta = delta
        input_t = _ternary(input_f, ctx.delta)
        ctx.save_for_backward(input_f)
        return input_t

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input_f, = ctx.saved_tensors
        grad_input = grad_output * (-1 <= input_f & input_f <= 1).float()
        return grad_input, None, None, None, None, None, None, None, None, None


class _ternary_py(torch.autograd.Function):
    @staticmethod
    def ternary_backward(grad_output: torch.Tensor, x: torch.Tensor, delta: float, order: int, threshold: float):
        scale = 2 * delta
        assert threshold <= scale
        tmp = torch.zeros_like(grad_output)
        # tmp += ((x < -threshold) | (x > threshold)).float() * slope
        # tmp += ((x >= -threshold) & (x < -scale)).float() * (order * ((x + scale) / -delta).pow(order - 1))
        # tmp += ((x >= -scale) & (x < -delta)).float() * (order * ((x + scale) / delta).pow(order - 1))
        # tmp += ((x >= -delta) & (x < 0)).float() * (order * (x / -delta).pow(order - 1))
        # tmp += ((x >= 0) & (x < delta)).float() * (order * (x / delta).pow(order - 1))
        # tmp += ((x >= delta) & (x < scale)).float() * (order * ((x - scale) / -delta).pow(order - 1))
        # tmp += ((x >= scale) & (x <= threshold)).float() * (order * ((x - scale) / delta).pow(order - 1))

        # tmp += ((x >= -threshold) & (x < -delta)).float() * order * ((x + scale) / delta).abs().pow(order - 1)
        # tmp += ((x >= -delta) & (x < delta)).float() * order * (x / delta).abs().pow(order - 1)
        # tmp += ((x >= delta) & (x <= threshold)).float() * order * ((x - scale) / delta).abs().pow(order - 1)

        tmp += ((x >= -threshold) & (x <= threshold)).float() * order * \
               (torch.fmod(x / delta + 3, 2) - 1).abs().pow(order - 1)
        return grad_output * tmp

    @staticmethod
    def forward(ctx, *inputs) -> torch.Tensor:
        input_f, running_delta, delta, momentum, training, ctx.order = inputs
        if momentum > 0:
            if training:
                ctx.delta = input_f.norm(1).item() * (delta / input_f.numel())  # = delta * |input_f|_1 / n
                running_delta.data = momentum * ctx.delta + (1.0 - momentum) * running_delta.data
            else:
                ctx.delta = running_delta.data.item()
        else:
            ctx.delta = delta
        input_t = _ternary(input_f, ctx.delta) * (2 * ctx.delta)
        ctx.save_for_backward(input_f)
        return input_t

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input_f, = ctx.saved_tensors
        grad_input = _ternary_py.ternary_backward(grad_output, input_f, ctx.delta, ctx.order, 2. * ctx.delta)
        return grad_input, None, None, None, None, None, None, None, None, None


def ternary(input_f: torch.Tensor, running_delta, delta, momentum, training, order, use_scale=True):
    if not use_scale:
        return _ternary_without_scale.apply(input_f, running_delta, delta, momentum, training)
    else:
        return _ternary_py.apply(input_f, running_delta, delta, momentum, training, order)


class Ternary(torch.nn.Module):
    def __init__(self, config: dict, *arg, **kwargs):
        super(Ternary, self).__init__()
        self.config = config
        self.delta = config.setdefault("delta", 0.5)
        self.momentum = config.setdefault("momentum", 0.01)
        self.track_running_stats = config.setdefault("track_running_stats", True)
        self.order = config.setdefault('order', 2)
        self.use_scale = config.setdefault('use_scale', True)
        assert self.momentum <= 1 and self.order > 0 and self.delta > 0
        self.register_buffer("running_delta", torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.momentum > 0:
            self.running_delta.fill_(self.delta * 0.7979)
        else:
            self.running_delta.fill_(self.delta)

    def forward(self, input_f):
        return ternary(input_f, self.running_delta, self.delta, self.momentum,
                       self.training and self.track_running_stats, self.order, self.use_scale)

    def extra_repr(self):
        return ", ".join(["{}={}".format(k, v) for k, v in self.config.items()])


if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 16)
    y = Ternary({})(x)
    print(y)
