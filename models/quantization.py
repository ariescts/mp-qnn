import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


OPTA = 'round'
OPTW = 'linear'


# binarization part

class ActBinarizeSTEv1(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ActBinarizeSTEv2(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ActBinarization(nn.Module):
    def __init__(self, type='sign'):
        super().__init__()
        assert type in ('sign', 'round')
        self.type = type

    def forward(self, x):
        if self.type == 'sign':
            return ActBinarizeSTEv1.apply(x)
        else:
            return ActBinarizeSTEv2.apply(x)


class WeightBinarizeSTE(Function):
    @staticmethod
    def forward(ctx, input, mean, channel_wise):
        if channel_wise:
            size = input.size()
            input = input.view(size[0], -1).transpose(1, 0)
            E = torch.mean(torch.abs(input), dim=0)
            output = torch.sign(input) * E
            output = output.transpose(1, 0).view(size).contiguous()
        elif mean:
            E = torch.mean(torch.abs(input))
            output = torch.sign(input) * E
        else:
            output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class WeightBinarization(nn.Module):
    def __init__(self, mean=True, channel_wise=False):
        super().__init__()
        self.mean = mean
        self.channel_wise = channel_wise

    def forward(self, w):
        return WeightBinarizeSTE.apply(w, self.mean, self.channel_wise)


# quantization part

def linear_quantization_params(num_bits, lower, upper, integral_zero_point=True):
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower).to(torch.float32)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper).to(torch.float32)

    lower = torch.min(lower, torch.zeros_like(lower))
    upper = torch.max(upper, torch.zeros_like(upper))

    n = 2 ** num_bits - 1
    diff = upper - lower

    alpha = diff / n
    beta = lower / alpha
    if integral_zero_point:
        beta.round_()
    return alpha, beta


def linear_quantize(input, alpha, beta, inplace=False):
    if inplace:
        input.div_(alpha).sub_(beta).round_()
        return input
    else:
        return torch.round( input / alpha - beta )


def linear_dequantize(input, alpha, beta, inplace=False):
    if inplace:
        input.add_(beta).mul_(alpha)
        return input
    else:
        return (input + beta) * alpha


class LinearQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, input, alpha, beta, dequantize, inplace):
        if inplace:
            ctx.make_dirty(input)
        output = linear_quantize(input, alpha, beta, inplace)
        if dequantize:
            output = linear_dequantize(output, alpha, beta, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class ActLinearQuantization(nn.Module):
    def __init__(self, num_bits, dequantize=True, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        self.dequantize = dequantize
        self.inplace = inplace
        self.alpha, self.beta = linear_quantization_params(num_bits, 0, 1, integral_zero_point=True)

    def forward(self, x):
        x_q = LinearQuantizeSTE.apply(x, self.alpha, self.beta, self.dequantize, self.inplace)
        return x_q


class WeightLinearQuantization(nn.Module):
    def __init__(self, num_bits, channel_wise=True, dequantize=True, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        self.channel_wise = channel_wise
        self.dequantize = dequantize
        self.inplace = inplace
        self.alpha = None
        self.beta = None

    def forward(self, w):
        if self.channel_wise:
            size = w.size()
            w = w.view(size[0], -1).transpose(1, 0)
            lower, _ = torch.min(w, dim=0)
            upper, _ = torch.max(w, dim=0)
            self.alpha, self.beta = linear_quantization_params(self.num_bits, lower, upper,
                                                               integral_zero_point=True)
            w_q = LinearQuantizeSTE.apply(w, self.alpha, self.beta, self.dequantize, self.inplace)
            w_q = w_q.transpose(1, 0).view(size).contiguous()
        else:
            lower = torch.min(w)
            upper = torch.max(w)
            self.alpha, self.beta = linear_quantization_params(self.num_bits, lower, upper,
                                                               integral_zero_point=True)
            w_q = LinearQuantizeSTE.apply(w, self.alpha, self.beta, self.dequantize, self.inplace)
        return w_q


class WeightDorefaQuantization(nn.Module):
    def __init__(self, num_bits, channel_wise=False, dequantize=True, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        self.channel_wise = channel_wise
        self.dequantize = dequantize
        self.inplace = inplace
        self.alpha, self.beta = linear_quantization_params(num_bits, 0, 1, integral_zero_point=True)

    def forward(self, w):
        w = torch.tanh(w)
        if self.channel_wise:
            size = w.size()
            w = w.view(size[0], -1).transpose(1, 0)
            M, _ = torch.max(torch.abs(w), dim=0)
            w = (w / M + 1) / 2
            w_q = LinearQuantizeSTE.apply(w, self.alpha, self.beta, self.dequantize, self.inplace)
            w_q = (w_q * 2 - 1).transpose(1, 0).view(size).contiguous()
        else:
            M = torch.max(torch.abs(w))
            w = (w/M + 1) / 2
            w_q = LinearQuantizeSTE.apply(w, self.alpha, self.beta, self.dequantize, self.inplace)
            w_q = w_q * 2 - 1
        return w_q


# network layer part

class Clamp(nn.Module):
    def __init__(self, upper=1):
        super(Clamp, self).__init__()
        self.upper = upper

    def forward(self, x):
        return F.hardtanh(x, min_val=0, max_val=self.upper, inplace=True) / self.upper


class Hardtanh6(nn.Module):
    def __init__(self, min_val=-1, max_val=1, inplace=True):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, x):
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace) / self.min_val


class QConv2d(nn.Conv2d):
    def __init__(self, ka, kw, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                 groups=1, bias=False, opta=OPTA, optw=OPTW):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride,
                                      dilation=dilation, groups=groups, bias=bias)
        self.opta = opta
        self.optw = optw
        self.weight_q = None
        assert 0 <= ka <= 32
        assert 0 <= kw <= 32
        self.ka = ka = int(ka)
        self.kw = kw = int(kw)

        if ka == 1:
            self.qa = ActBinarization(type=opta)
        elif ka == 32:
            self.qa = lambda x: x
        else:
            self.qa = ActLinearQuantization(ka)

        if kw == 1:
            self.qw = WeightBinarization(channel_wise=True)
        elif kw == 32:
            self.qw = lambda x: x
        else:
            if optw == 'linear':
                self.qw = WeightLinearQuantization(kw)
            elif optw == 'dorefa':
                self.qw = WeightDorefaQuantization(kw)
            else:
                raise ValueError('Unsupported quantization method.')

    def forward(self, x):
        x_q = self.qa(x)
        self.weight_q = self.qw(self.weight)
        out = F.conv2d(x_q, self.weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QLinear(nn.Linear):
    def __init__(self, ka, kw, in_features, out_features, bias=False, opta=OPTA, optw=OPTW):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.opta = opta
        self.optw = optw
        self.weight_q = None
        assert 0 <= ka <= 32
        assert 0 <= kw <= 32
        self.ka = ka = int(ka)
        self.kw = kw = int(kw)

        if ka == 1:
            self.qa = ActBinarization(type=opta)
        elif ka == 32:
            self.qa = lambda x: x
        else:
            self.qa = ActLinearQuantization(ka)

        if kw == 1:
            self.qw = WeightBinarization(channel_wise=True)
        elif kw == 32:
            self.qw = lambda x: x
        else:
            if optw == 'linear':
                self.qw = WeightLinearQuantization(kw)
            elif optw == 'dorefa':
                self.qw = WeightDorefaQuantization(kw)
            else:
                raise ValueError('Unsupported quantization method.')

    def forward(self, x):
        x_q = self.qa(x)
        self.weight_q = self.qw(self.weight)
        out = F.linear(x_q, self.weight_q, self.bias)
        return out


# test part

if __name__ == '__main__':
    torch.cuda.manual_seed(100)
    torch.manual_seed(100)

    x1 = torch.rand(2, 10)
    linear = QLinear(ka=1, kw=2, in_features=10, out_features=20)
    y1 = linear(x1)
    print(linear.weight_q[0])
    print(y1[0])

    x2 = torch.rand(2, 3, 32, 32)
    conv = QConv2d(ka=1, kw=2, in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1)
    y2 = conv(x2)
    print(conv.weight_q[0])
    print(y2[0, :, 0, 0])
