"""
@Brief: Quantization and Dequantization for the output of neural networks.
"""

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Num2Bit(Num, B):
    """
    :brief：Nums to bits.
    :param Num：Num tensor list.
    :param B: Bits number of quantization.
    :return：Quantization results.
    """
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = bit.reshape(-1, Num_.shape[1] * B)  # (batch, )

    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    """
    :brief：Bits to nums.
    :param Bit：Bit tensor list.
    :param B: Bits number of quantization.
    :return：Dequantization results.
    """
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])

    num = torch.zeros(Bit_[:, :, 1].shape).to(device)
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)

    return num


class Quantization(torch.autograd.Function):
    """
    :brief：This part implement the quantization operations.
    :param torch.autograd.Function：Define your own layer, you must rewrite forward and backward function. This class is
    like tf.keras.layers.Layer.
    :param ctx: This is the autograd calculated by backward function.
    :param x: Input tensor x.
    :param B: Bits number of quantization.
    :return：Quantizationed bits tensor.
    """
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)  # 量化时，直接将每一个float量化成的比特和作为梯度
        return grad_num, None


class Dequantization(torch.autograd.Function):
    """
    :brief：This part implement the dequantization operations.
    :param torch.autograd.Function：Define your own layer, you must rewrite forward and backward function. This class is
    like tf.keras.layers.Layer.
    :param ctx: This is the autograd calculated by backward function.
    :param x: Input tensor x.
    :param B: Bits number of quantization.
    :return：Object.
    """
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)  # 解量化时，直接将解量化的结果作为梯度值
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):
    """
    :brief：QuantizationLayer from nn.Module.
    :param B：Bits number of quantization.
    :return：Object.
    """
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)

        return out


class DequantizationLayer(nn.Module):
    """
    :brief：DequantizationLayer from nn.Module.
    :param B：Bits number of quantization.
    :return：Object.
    """
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


if __name__ == "__main__":
    # Example
    quantization = QuantizationLayer(4)  # 4 bits quantization
    dequantization = DequantizationLayer(4)

    # x = torch.randn([1, 196, 1, 96]).to(device)
    # x[0][0][0][0] = 0.1
    x = torch.randn(1, 18816).to(device)
    # print(x.shape)
    y = quantization(x).cuda()
    print(y)
    z = dequantization(y).cuda()
    # print(z.shape)
    # z = z.reshape(-1, 196, 1, 96)


