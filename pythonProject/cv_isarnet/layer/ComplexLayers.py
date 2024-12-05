import torch
from torch.nn import Flatten
from torch.nn import Module
from torch.nn import Conv2d, Conv1d
from torch.nn import Linear
from torch.nn import ConvTranspose2d
from torch.nn.functional import dropout
from torch.nn.functional import dropout2d


def apply_complex_conv(fr, fi, input, dtype=torch.complex64):
    """
    fr(input.real)：The real part of the convolution kernel * The real part of the input。
    fi(input.imag)：The  imaginary part of the convolution kernel * The  imaginary part of the input
    fr(input.imag)：The real part of the convolution kernel * The  imaginary part of the input
    fi(input.real)：The  imaginary part of the convolution kernel * The real part of the input
    """
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexConv1d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ComplexConv1d, self).__init__()
        """
        ComplexConv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode='zeros')
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        """
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, inp):
        return apply_complex_conv(self.conv_r, self.conv_i, inp)


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        """
        ComplexConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        """
        # real & image part
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex_conv(self.conv_r, self.conv_i, input)


class ComplexConv3d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ComplexConv3d, self).__init__()
        """
        ComplexConv3d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        bias=True, padding_mode='zeros'):
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to all six sides of the input. Default: 0
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        """
        # real & image part
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, inp):
        return apply_complex_conv(self.conv_r, self.conv_i, inp)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        """
        ComplexLinear(in_features, out_features, bias=True)
        in_features (int) – size of each input sample
        out_features (int) – size of each output sample
        bias (bool) – If set to False, the layer will not learn an additive bias. Default: True
        """
        # real & image part
        self.fc_r = Linear(in_features, out_features, bias)
        self.fc_i = Linear(in_features, out_features, bias)

    def forward(self, input):
        return apply_complex_conv(self.fc_r, self.fc_i, input)


class ComplexFlatten(Flatten):

    def __init__(self, start_dim=0, end_dim=-1):
        super(ComplexFlatten, self).__init__()
        """
        ComplexFlatten(start_dim=0, end_dim=-1)
        input (Tensor) – the input tensor.
        start_dim (int) – the first dim to flatten
        end_dim (int) – the last dim to flatten
        """
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        real_flat = torch.flatten(input.real, self.start_dim, self.end_dim)
        imag_flat = torch.flatten(input.imag, self.start_dim, self.end_dim)
        return real_flat.type(torch.complex64) + 1j * imag_flat.type(torch.complex64) # Keep input dtype


class ComplexConvTransposed2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1):
        super(ComplexConvTransposed2d, self).__init__()
        """
        ComplexConvTransposed2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        output_padding=0, groups=1, bias=True, dilation=1)
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – dilation * (kernel_size - 1) - padding zero-padding will be added to both 
        sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional) – Additional size added to one side of each dimension 
        in the output shape. Default: 0
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        """
        # real # image part
        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation)

    def forward(self, input):
        return apply_complex_conv(self.conv_tran_r, self.conv_tran_i, input)


"""
input1 = torch.randn(2, 1, 3, 3) + 1j * 2 * torch.rand(2, 1, 3, 3)
print(f"Initial:{input1}")
a = ComplexFlatten()
x = a(input1)
print(f"Flatten:{x}")
"""

