import torch
from torch.nn import Module, Parameter, init
import torch.nn.functional as F


def complex_dropout(inp, p=0.5, training=True, inplace=False):
    if inp.is_cuda:
        mask = torch.ones(*inp.shape, dtype=torch.float32)
        mask = F.dropout(mask, p, training, inplace) * 1 / (1 - p)
        # This is equivalent to both the real part and the imaginary part making a dropout with probability p
        # *1/(1-p) again
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to(device).type(inp.dtype)
        return mask * inp
    else:
        mask = torch.ones(*inp.shape, dtype=torch.float32)
        mask = F.dropout(mask, p, training, inplace) * 1 / (1 - p)
        mask.type(inp.dtype)
        return mask * inp
# need to have the same dropout mask for real and imaginary part


def complex_dropout_respectively(inp, p=0.5, training=True, inplace=False):
    if inp.is_cuda:
        mask_r = torch.ones(*inp.shape, dtype=torch.float32)
        mask_r = F.dropout(mask_r, p, training, inplace) * 1 / (1 - p)
        mask_i = torch.ones(*inp.shape, dtype=torch.float32)
        mask_i = F.dropout(mask_i, p, training, inplace) * 1 / (1 - p)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_r = mask_r.to(device).type(inp.dtype)
        mask_i = mask_i.to(device).type(inp.dtype)
        return mask_r * inp.real + 1j * mask_i * inp.imag
    else:
        mask_r = torch.ones(*inp.shape, dtype=torch.float32)
        mask_r = F.dropout(mask_r, p, training, inplace) * 1 / (1 - p)
        mask_i = torch.ones(*inp.shape, dtype=torch.float32)
        mask_i = F.dropout(mask_i, p, training, inplace) * 1 / (1 - p)
        mask_r = mask_r.type(inp.dtype)
        mask_i = mask_i.type(inp.dtype)
        return mask_r * inp.real + 1j * mask_i * inp.imag
# dropout of real part and virtual part is regarded as adding noise interference


def complex_dropout2d(inp, p=0.5, training=True, inplace=False):
    if inp.is_cuda:
        mask = torch.ones(*inp.shape, dtype=torch.float32)
        mask = F.dropout2d(mask, p, training, inplace) * 1 / (1 - p)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to(device).type(inp.dtype)
        return mask * inp
    else:
        mask = torch.ones(*inp.shape, dtype=torch.float32)
        mask = F.dropout2d(mask, p, training, inplace) * 1 / (1 - p)
        mask.type(inp.dtype)
        return mask * inp
# Randomly zero out entire channels (a channel is a 2D feature map).


class ComplexDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout, self).__init__()
        """
        p (float) – probability of an element to be zeroed. Default: 0.5
        training (bool) – apply dropout if is True. Default: True
        inplace (bool) – If set to True, will do this operation in-place. Default: False
        """
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, inp):
        if self.training:
            return complex_dropout(inp, p=self.p, inplace=self.inplace)
        else:
            return inp


class ComplexDropout2D(Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout2D, self).__init__()
        """
        p (float) – probability of a channel to be zeroed. Default: 0.5
        training (bool) – apply dropout if is True. Default: True
        inplace (bool) – If set to True, will do this operation in-place. Default: False
        """
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, p=self.p, inplace=self.inplace)
        else:
            return inp


class ComplexDropoutRespectively(Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropoutRespectively, self).__init__()
        """
        p (float) – probability of an element to be zeroed. Default: 0.5
        training (bool) – apply dropout if is True. Default: True
        inplace (bool) – If set to True, will do this operation in-place. Default: False
        """
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, inp):
        if self.training:
            return complex_dropout_respectively(inp, p=self.p, inplace=self.inplace)
        else:
            return inp


"""
# Simple verification of ComplexDropout
input1 = torch.rand(1, 1, 3, 3, 3) + 1j * 2 * torch.rand(1, 1, 3, 3, 3)
print(f"Initial:{input1}")
m = ComplexDropout(p=0.5)
output1 = m(input1)
print(f"Dropout:{output1}")
"""
"""
# Simple verification of ComplexDropout2D
input1 = torch.rand(1, 3, 1, 2, 2) + 1j * 2 * torch.rand(1, 3, 1, 2, 2)
print(f"Initial:{input1}")
m = ComplexDropout2D(p=0.5)
output1 = m(input1)
print(f"Dropout:{output1}")
"""
"""
# Simple verification of ComplexDropoutRespectively
input1 = torch.rand(1, 1, 3, 3, 3) + 1j * 2 * torch.rand(1, 1, 3, 3, 3)
print(f"Initial:{input1}")
m = ComplexDropoutRespectively(p=0.5)
output1 = m(input1)
print(f"Dropout:{output1}")
"""
