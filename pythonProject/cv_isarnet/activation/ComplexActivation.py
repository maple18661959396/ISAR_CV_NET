import torch
import torch.nn.functional as F
from typing import Optional, Callable
import math

"""
This module contains many complex-valued activation functions to be used by CVNN class.
"""


def complex_relu(inp):
    return F.relu(inp.real).type(torch.complex64) + 1j*F.relu(inp.imag).type(torch.complex64)
# Apply the ReLU activation function to both the real and imaginary parts separately.


def complex_elu(inp):
    return F.elu(inp.real).type(torch.complex64) + 1j * F.elu(inp.imag).type(torch.complex64)
# Apply the elu activation function to both the real and imaginary parts separately.

def complex_exponential(inp):
    return torch.exp(inp.real).type(torch.complex64) + 1j * torch.exp(inp.imag).type(torch.complex64)
# Apply the exp activation function to both the real and imaginary parts separately.

"""
def complex_sigmoid1(inp):
    return F.sigmoid(inp.real).type(torch.complex64) + 1j * F.sigmoid(inp.imag).type(torch.complex64)
# Apply the sigmoid activation function to both the real and imaginary parts separately.
"""


def complex_sigmoid(inp):
    return torch.sigmoid(inp.real).type(torch.complex64) + 1j * torch.sigmoid(inp.imag).type(torch.complex64)
# F.sigmoid has been deprecated, use torch.sigmoid.


"""
def complex_tanh(inp):
    return F.tanh(inp.real).type(torch.complex64) + 1j * F.tanh(inp.imag).type(torch.complex64)
# Apply the tanh activation function to both the real and imaginary parts separately.
"""


def complex_tanh(inp):
    return torch.tanh(inp.real).type(torch.complex64) + 1j * torch.tanh(inp.imag).type(torch.complex64)
# F.tanh has been deprecated, use torch.tanh.


def complex_hard_sigmoid(inp):
    return F.hardsigmoid(inp.real).type(torch.complex64) + 1j * F.hardsigmoid(inp.imag).type(torch.complex64)
# Apply the hard_sigmoid activation function to both the real and imaginary parts separately.


def complex_leaky_relu(inp):
    return F.leaky_relu(inp.real).type(torch.complex64) + 1j * F.leaky_relu(inp.imag).type(torch.complex64)
# Apply the leaky_relu activation function to both the real and imaginary parts separately.

def complex_selu(inp):
    return F.selu(inp.real).type(torch.complex64) + 1j * F.selu(inp.imag).type(torch.complex64)
# Apply the selu activation function to both the real and imaginary parts separately.

def complex_softplus(inp):
    return F.softplus(inp.real).type(torch.complex64) + 1j * F.softplus(inp.imag).type(torch.complex64)
# Apply the softplus activation function to both the real and imaginary parts separately.

def complex_softsign(inp):
    return F.softsign(inp.real).type(torch.complex64) + 1j * F.softsign(inp.imag).type(torch.complex64)
# Apply the softsign activation function to both the real and imaginary parts separately.

def complex_softmax(inp, axis=-1):
    return F.softmax(inp.real, axis).type(torch.complex64) + 1j * F.softmax(inp.imag, axis).type(torch.complex64)
# Apply the softmax activation function to both the real and imaginary parts separately.

def modrelu(inp, b: float = 1., c: float = 1e-3):
    """
    mod ReLU presented in "Unitary Evolution Recurrent Neural Networks"
        from M. Arjovsky et al. (2016)
        URL: https://arxiv.org/abs/1511.06464
    A variation of the ReLU named modReLU. It is a pointwise nonlinearity,
    modReLU(inp) : C -> C, which affects only the absolute
    value of a complex number, defined:
        modReLU(inp) = ReLU(|inp|+b)*inp/|inp|
    TODO: See how to check the non zero abs.
    """
    abs_inp = torch.abs(inp)
    return F.relu(abs_inp + b).type(torch.complex64) * inp / (abs_inp + c).type(torch.complex64)
# modReLU(z) = ReLU(|z|+b)*z/|z|


def zrelu(inp, epsilon=1e-7):
    """
    zReLU presented in "On Complex Valued Convolutional Neural Networks"
        from Nitzan Guberman (2016).
    This methods let's the output as the input if both real and imaginary parts are positive.

    https://stackoverflow.com/questions/49412717/advanced-custom-activation-function-in-keras-tensorflow
    """
    inp_re = F.relu(inp.real)
    inp_im = F.relu(inp.imag)
    out_re = inp_re * inp_im / (inp_im + epsilon)
    out_im = inp_re * inp_im / (inp_re + epsilon)
    return out_re.type(torch.complex64) + 1j * out_im.type(torch.complex64)
# Let the output be the input if both the real and imaginary parts are positive.


def complex_cardioid(inp):
    """
    Complex cardioid presented in "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting"
        from V. Patrick (2017).

    This function maintains the phase information while attenuating the magnitude based on the phase itself.
    For real-valued inputs, it reduces to the ReLU.
    """
    return (torch.cos(torch.angle(inp)) + 1).type(torch.complex64) * inp / 2
# This function maintains the phase information while attenuating the magnitude based on the phase itself.


def sigmoid_real(inp):
    return torch.sigmoid(inp.real + inp.imag)
# Use the sigmoid activation function for the sum of the real and imaginary parts


def softmax_real_with_abs(inp, axis=-1):
    """
    Applies the softmax function to the modulus of inp.
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
        return F.softmax(torch.abs(inp), axis)
    else:
        return F.softmax(inp, axis)
# Use the sigmoid activation function for |inp|


def softmax_real_with_avg(inp, axis=-1):
    """
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
        return 0.5*(F.softmax(inp.real, axis) + 0.5*F.softmax(inp.imag, axis))
    else:
        return F.softmax(inp, axis)
# Use the softmax activation function in the real and imaginary parts, respectively.Then take the average


def softmax_real_with_mult(inp, axis=-1):
    """
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
         return F.softmax(inp.real, axis) * F.softmax(inp.imag, axis)
    else:
        return F.softmax(inp, axis)
# Apply the softmax activation function to both the real and imaginary parts separately. Then, multiply the values


def softmax_of_softmax_real_with_mult(inp, axis=-1):
    """
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
        return F.softmax((F.softmax(inp.real, axis) * F.softmax(inp.imag, axis)), axis)
    else:
        return F.softmax(inp, axis)
# Use the softmax_real_with_mult activation function then use the softmax activation.


def softmax_of_softmax_real_with_avg(inp, axis=-1):
    """
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
        return F.softmax(0.5*(F.softmax(inp.real, axis) + 0.5*F.softmax(inp.imag, axis)), axis)
    else:
        return F.softmax(inp, axis)
# Use the softmax_real_with_avg activation function then use the softmax activation.


def softmax_real_by_parameter(inp, axis=-1, params: Optional[dict] = None):
    if params is None:
        params = {
            'abs': True,
            'angle': True,
            'real': True,
            'imag': True
        }
    result = []
    for k in params:
        if k == 'abs':
            result.append(F.softmax(torch.abs(inp), axis))
        if k == 'angle':
            result.append(F.softmax(torch.angle(inp), axis))
        if k == 'real':
            result.append(F.softmax(inp.real, axis))
        if k == 'imag':
            result.append(F.softmax(inp.imag, axis))
    return torch.stack(result)
# Use the softmax activation function according to parameters


def softmax_real_with_polar(inp, axis=-1):
    """
    The softmax activation function transforms the outputs so that all values are in range (0, 1) and sum to 1.
    It is often used as the activation for the last layer of a classification network because the result could be
    interpreted as a probability distribution.
    :param inp: Input tensor.
    :return: Real-valued tensor of the applied activation function
    """
    if inp.dtype.is_complex:
        return 0.5 * (F.softmax(torch.abs(inp), axis) + 0.5 * F.softmax(torch.angle(inp), axis))
    else:
        return F.softmax(inp, axis)
# Use the softmax_real_with_avg activation function for polar


def georgiou_cdbp(inp, r: float = 1, c: float = 1e-3):
    """
    Activation function proposed by G. M. Georgioy and C. Koutsougeras in
        https://ieeexplore.ieee.org/abstract/document/142037
    """
    return inp / (c + torch.abs(inp) / r).type(torch.complex64)
# Complex Backpropagation Activation Function


def complex_signum(inp, k: Optional[int] = None):
    """
    Complex signum activation function is very similar to mvn_activation.
    For a detailed explanation refer to:
        https://ieeexplore.ieee.org/abstract/document/548176
    """
    if k:
        # values = np.linspace(pi / k, 2 * pi - pi / k, k)
        angle_cast = torch.floor(torch.angle(inp) * k / (2 * math.pi))
        # import pdb; pdb.set_trace()
        return torch.exp(
            torch.zeros(torch.tensor.inp.shape, dtype=torch.complex64)+ 1j *(angle_cast + 0.5) * 2 * math.pi / k)
    else:
        return torch.exp(torch.zeros(inp.shape, dtype=torch.complex64) + 1j * torch.angle(inp))


def mvn_activation(inp, k: Optional[int] = None):
    """
    Function inspired by Naum Aizenberg.
        A multi-valued neuron (MVN) is a neural element with n inputs and one output lying on the unit circle,
        and with complex-valued weights.
    Works:
        https://link.springer.com/article/10.1007%2FBF01068667
        http://pefmath2.etf.rs/files/93/399.pdf
    """
    if k:
        # values = np.linspace(pi / k, 2 * pi - pi / k, k)
        angle_cast = torch.floor(torch.angle(inp) * k / (2 * math.pi))
        # import pdb; pdb.set_trace()
        return torch.exp(
            torch.zeros(inp.shape, dtype=torch.complex64) + 1j * (angle_cast + 0.5) * 2 * math.pi / k)
    else:
        return torch.exp(torch.zeros(inp.shape, dtype=torch.complex64) + 1j * torch.angle(inp))


# Polar form
def apply_pol(inp, amp_fun: Callable[[torch.tensor], torch.tensor], pha_fun: Optional[Callable[[torch.tensor], torch.tensor]] = None):
    amp = amp_fun(torch.abs(inp))
    pha = torch.angle(inp)
    if pha_fun is not None:
        pha = pha_fun(pha)
    return (amp * torch.cos(pha)).type(torch.complex64) + 1j * (amp * torch.sin(pha)).type(torch.complex64)
#  polar form to complex number form


def pol_tanh(inp):
    return apply_pol(inp, torch.tanh)
# Use the tanh activation function


def pol_sigmoid(inp):
    return apply_pol(inp, torch.sigmoid)
# Use the sigmoid activation function


def pol_selu(inp):
    r_0 = torch.abs(inp)
    r_1 = F.selu(r_0)
    return (inp.real * r_1 / r_0).type(torch.complex64) + 1j * (inp.imag * r_1 / r_0).type(torch.complex64)
# Use the selu activation function


"""
# test
input1 = torch.randn(2, 1, 3, 3) + 1j * 2 * torch.rand(2, 1, 3, 3)
print(f"Initial tensor:{input1}")
x = softmax_real_with_polar(input1)
print(f"Activation tensor:{x}")
"""


