import torch
'''
Implementation of convolution
'''


def apply_complex_conv(fr, fi, input, dtype=torch.complex64):
    """
    fr(input.real)：The real part of the convolution kernel * The real part of the input。
    fi(input.imag)：The  imaginary part of the convolution kernel * The  imaginary part of the input
    fr(input.imag)：The real part of the convolution kernel * The  imaginary part of the input
    fi(input.real)：The  imaginary part of the convolution kernel * The real part of the input
    """
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)