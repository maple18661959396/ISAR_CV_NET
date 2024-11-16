import torch
from torch.nn import Module, Parameter, init
import torch.nn.functional as F


def complex_avg_pool1d(inp, kernel_size, stride=None, padding=0, ceil_mode=False,count_include_pad=True):
    absolute_value_real = F.avg_pool1d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad)
    absolute_value_imag = F.avg_pool1d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad)
    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64)
# Use the torch.nn.functional.avg_pool2d function in the real and imaginary parts, respectively.


def complex_avg_pool2d(inp, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True, divisor_override=None):
    absolute_value_real = F.avg_pool2d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                       divisor_override=divisor_override)
    absolute_value_imag = F.avg_pool2d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                       divisor_override=divisor_override)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64)
# Use the torch.nn.functional.avg_pool2d function in the real and imaginary parts, respectively.


def complex_avg_pool3d(inp, kernel_size, stride=None, padding=0, ceil_mode=False,
                       count_include_pad=True, divisor_override=None):
    absolute_value_real = F.avg_pool3d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                       divisor_override=divisor_override)
    absolute_value_imag = F.avg_pool3d(inp.real, kernel_size=kernel_size, stride=stride, padding=padding,
                                       ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                       divisor_override=divisor_override)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64)
# Use the torch.nn.functional.avg_pool3d function in the real and imaginary parts, respectively.


def complex_polar_avg_pooling2d(inp, kernel_size, stride=None, padding=0, ceil_mode=False,
                                count_include_pad=True, divisor_override=None):
    inputs_abs = torch.abs(inp)
    output_abs = F.avg_pool2d(inputs_abs, kernel_size=kernel_size, stride=stride, padding=padding,
                              ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                              divisor_override=divisor_override)
    inputs_angle = torch.angle(inp)
    avg_unit_x = F.avg_pool2d(torch.cos(inputs_angle), kernel_size=kernel_size, stride=stride, padding=padding,
                              ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                              divisor_override=divisor_override)
    avg_unit_y = F.avg_pool2d(torch.sin(inputs_angle), kernel_size=kernel_size, stride=stride, padding=padding,
                              ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                              divisor_override=divisor_override)
    output_angle = torch.angle(avg_unit_x.type(torch.complex64) + 1j * avg_unit_y.type(torch.complex64))
    if inp.dtype.is_complex:
        output = (output_abs * torch.cos(output_angle)).type(torch.complex64) + 1j * (output_abs * torch.sin(output_angle)).type(torch.complex64)
    else:
        output = output_abs
    return output
# AvgPooling under polar coordinates input


class ComplexAvgPool1D(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(ComplexAvgPool1D, self).__init__()
        """
        ComplexAvgPool1D(kernel_size , stride=None , padding=0 , ceil_mode=False ,
        count_include_pad=True)
        kernel_size (Union[int, Tuple[int]]) – the size of the window
        stride (Union[int, Tuple[int]]) – the stride of the window. Default value is kernel_size
        padding (Union[int, Tuple[int]]) – implicit zero padding to be added on both sides
        ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
        count_include_pad (bool) – when True, will include the zero-padding in the averaging calculation
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, inp):
        return complex_avg_pool1d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad)


class ComplexAvgPool2D(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2D, self).__init__()
        """
        ComplexAvgPool2D(kernel_size , stride=None , padding=0 , ceil_mode=False ,
        count_include_pad=True , divisor_override=None )
        kernel_size (Union[int, Tuple[int, int]]) – the size of the window
        stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
        padding (Union[int, Tuple[int, int]]) – implicit zero padding to be added on both sides
        ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
        count_include_pad (bool) – when True, will include the zero-padding in the averaging calculation
        divisor_override (Optional[int]) – if specified, it will be used as divisor, otherwise size of the pooling region will be used.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexAvgPool3D(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool3D, self).__init__()
        """
        ComplexAvgPool3D(kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True, divisor_override=None)
        kernel_size (Union[int, Tuple[int, int, int]]) – the size of the window
        stride (Union[int, Tuple[int, int, int]]) – the stride of the window. Default value is kernel_size
        padding (Union[int, Tuple[int, int, int]]) – implicit zero padding to be added on all three sides
        ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
        count_include_pad (bool) – when True, will include the zero-padding in the averaging calculation
        divisor_override (Optional[int]) – if specified, it will be used as divisor, otherwise kernel_size will be used
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool3d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexPolarAvgPooling2D(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(ComplexPolarAvgPooling2D, self).__init__()
        """
        AvgPooling2d under polar coordinates input, and  parameter Settings are the same as ComplexAvgPool2D
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_polar_avg_pooling2d(inp, kernel_size=self.kernel_size,
                                           stride=self.stride, padding=self.padding,
                                           ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                           divisor_override=self.divisor_override)


def _retrieve_elements_from_indices(tensor, indices):
    # Retrieve from the indices
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complex_max_pool2d(
        inp, kernel_size, stride=None, padding=0, dilation=1,
        ceil_mode=False, return_indices=False
):
    absolute_value, indices = F.max_pool2d(
        inp.abs(), kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=True,
    )
    """
    # retrieve the corresponding real and imaginary parts by the index of the maximum modular value
    pool_output_r = _retrieve_elements_from_indices(inp.real, indices)
    pool_output_i = _retrieve_elements_from_indices(inp.imag, indices)
    return pool_output_r.type(torch.complex64) + 1j * pool_output_i.type(torch.complex64)
    """
    absolute_value = absolute_value.type(torch.complex64)
    # Record the maximum modular
    angle = torch.atan2(inp.imag, inp.real)
    # Record the Angle of each point
    angle = _retrieve_elements_from_indices(angle, indices)
    # Use the index to retrieve the corresponding phase value
    if return_indices is False:
        return absolute_value * (torch.cos(angle).type(torch.complex64)
                                 + 1j * torch.sin(angle).type(torch.complex64))
    else:
        return absolute_value * (torch.cos(angle).type(torch.complex64)
                                 + 1j * torch.sin(angle).type(torch.complex64)), indices
# Use the torch.nn.functional.max_pool2d function in the module of the complex number


class ComplexMaxPool2D(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, ceil_mode=False, return_indices=False):
        super(ComplexMaxPool2D, self).__init__()
        """
        ComplexMaxPool2D(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
        kernel_size – size of the pooling region. Can be a single number or a tuple (kH, kW)
        stride – stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size
        padding – Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation – The stride between elements within a sliding window, must be > 0.
        ceil_mode – If True, will use ceil instead of floor to compute the output shape. This ensures that every element
        in the input tensor is covered by a sliding window.
        return_indices – If True, will return the argmax along with the max values. Useful for
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)


def complex_un_pool2d(inp, indices, kernel_size, stride=None, padding: torch.tensor = 0, output_size=None):
    absolute_value_real = F.max_unpool2d(inp.real, indices=indices, kernel_size=kernel_size,
                                         stride=stride, padding=padding, output_size=output_size)
    absolute_value_imag = F.max_unpool2d(inp.imag, indices=indices, kernel_size=kernel_size,
                                         stride=stride, padding=padding, output_size=output_size)
    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64)
# Use the torch.nn.functional.max_unpool2d function in the real and imaginary parts, respectively.


class ComplexUnPooling2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexUnPooling2D, self).__init__()
        """
        ComplexUnPooling2D(kernel_size, stride=None, padding=0)
        kernel_size (int or tuple) – Size of the max pooling window.
        stride (int or tuple) – Stride of the max pooling window. It is set to kernel_size by default.
        padding (int or tuple) – Padding that was added to the input
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, inp, indices, output_size=None):
        return complex_un_pool2d(inp, indices, kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding, output_size=output_size)


"""
# Simple verification of MaxPool2D and UnPooling2D
input1 = torch.rand(1, 1, 4, 4) + 1j * 2 * torch.rand(1, 1, 4, 4)
print(f"Initial:{input1}")
pool = ComplexMaxPool2D(2, stride=2, return_indices=True)
unpool = ComplexUnPooling2D(2, stride=2)
output1,mo indices = pool(input1)
print(f"pooling:{output1}")
print(f"indices:{indices}")
output2 = unpool(output1, indices)
print(f"unpool:{output2}")
"""
'''
# Simple verification of ComplexAvgPooling3D
input1 = torch.rand(4, 4, 4, 4, 4) + 1j * 2 * torch.rand(4, 4, 4, 4, 4)
print(f"Initial:{input1}")
pool = ComplexAvgPool3D((2, 2, 2), stride=(2, 2, 2))
output1 = pool(input1)
print(f"pooling{output1}")
'''
'''
# Simple verification of ComplexAvgPool1D
input1 = torch.rand(1, 1, 10) + 1j * 2 * torch.rand(1, 1, 10)
print(f"Initial:{input1}")
pool = ComplexAvgPool1D(2, stride=2)
output1 = pool(input1)
print(f"pooling:{output1}")
'''
'''
# Simple verification of ComplexPolarAvgPooling2D and ComplexAvgPool2D
input1 = torch.rand(2, 1, 4, 4) + 1j * 2 * torch.rand(2, 1, 4, 4)
print(f"Initial:{input1}")
pool1 = ComplexPolarAvgPooling2D([2, 2])
pool2 = ComplexAvgPool2D([2, 2])
output1 = pool1(input1)
print(f"ComplexPolarAvgPooling2D:{output1}")
output2 = pool2(input1)
print(f"ComplexAvgPool2D:{output2}")
'''