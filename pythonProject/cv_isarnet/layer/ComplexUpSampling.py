import torch
from torch.nn import Module, Parameter, init
import torch.nn.functional as F


def complex_up_sampling(inp, size=None, scale_factor=None, mode='nearest', align_corners=None):
    absolute_value_real = F.interpolate(inp.real, size, scale_factor, mode, align_corners)
    absolute_value_imag = F.interpolate(inp.imag, size, scale_factor, mode, align_corners)
    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)
# Use the torch.nn.functional.interpolate function in the real and imaginary parts, respectively.


class ComplexUpSampling(Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(ComplexUpSampling, self).__init__()
        """
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – output spatial size.
        scale_factor (float or Tuple[float]) – multiplier for spatial size. If scale_factor is a tuple, its length has
        to match the number of spatial dimensions; input.dim() - 2.
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' |
        | 'area' | 'nearest-exact'. Default: 'nearest'
        align_corners (bool, optional) – Geometrically, we consider the pixels of the input and output as squares rather
        than points. If set to True, the input and output tensors are aligned by the center points of their corner 
        pixels, preserving the values at the corner pixels. If set to False, the input and output tensors are aligned 
        by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary
        values, making this operation independent of input size when scale_factor is kept the same. This only has an 
        effect when mode is 'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: False
        """
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, inp):
        return complex_up_sampling(inp, size=self.size, scale_factor=self.scale_factor,
                                   mode=self.mode, align_corners=self.align_corners)


class ComplexUpSamplingBilinear2d(ComplexUpSampling):
    # mode='bilinear' specifically for 2D
    def __init__(self, size=None, scale_factor=None):
        super(ComplexUpSampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, inp):
        return complex_up_sampling(inp, size=self.size, scale_factor=self.scale_factor,
                                   mode='bilinear', align_corners=True)


class ComplexUpSamplingNearest2d(ComplexUpSampling):
    # mode='nearest' specifically for 2D
    def __init__(self, size=None, scale_factor=None):
        super(ComplexUpSampling, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, inp):
        return complex_up_sampling(inp, size=self.size, scale_factor=self.scale_factor, mode='nearest')


"""
input1 = torch.rand(1, 1, 2, 2) + 1j * 2 * torch.rand(1, 1, 2, 2)
# Simple verification of ComplexUpSampling
print(f"Initial:{input1}")
interpolate1 = ComplexUpSampling(scale_factor=2)
output1 = interpolate1(input1)
print(f"UpSampling:{output1}")
"""
"""
input1 = torch.rand(1, 1, 2, 2) + 1j * 2 * torch.rand(1, 1, 2, 2)
# Simple verification of ComplexUpSampling （mode='bilinear'）
print(f"Initial:{input1}")
interpolate1 = ComplexUpSampling(scale_factor=2, mode='bilinear', align_corners=False)
interpolate2 = ComplexUpSampling(scale_factor=2, mode='bilinear', align_corners=True)
output1 = interpolate1(input1)
output2 = interpolate2(input1)
print(f"UpSampling(align_corners=False):{output1}")
print(f"UpSampling（align_corners=True）:{output2}")
"""
"""
input1 = torch.rand(1, 1, 3) + 1j * 2 * torch.rand(1, 1, 3)
# Simple verification of ComplexUpSampling （mode='linear'）
print(f"Initial:{input1}")
interpolate1 = ComplexUpSampling(size=6, mode='linear')
output1 = interpolate1(input1)
print(f"UpSampling:{output1}")
"""