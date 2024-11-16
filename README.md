# Complex-valued neural networks

---

## Introduction

The idea of this library is just to implement Complex layers, so that everything else stays the same as any PyTorch code.
I am basically working with Complex-Valued Neural Networks.  In the need of making our coding more dynamic we build this
library not to have to repeat the same code over and over and accelerate the speed of coding.

---

## Instructions for use

Before version 1.7 of PyTorch, complex tensor were not supported.  Since version 1.7, complex tensors of type torch.complex64 are allowed,
but only a limited number of operation are supported.  (hence requires PyTorch version >= 1.7)

---

## Installation

### Using [PIP]

Only use complex-valued neural networks library:
`pip install cv-net-library`

### Using GitHub

Useful if you want to modify the source code and view the relevant tests:
`address`

---

## library Function

The idea of this library is that when the network input is complex-valued, you can use this library function to 
make it possible to build the network directly in a manner similar to the real number network.  
All functions already work Ok for complex inputs, so nothing should change.

The functions in the Complex-valued neural networks has the same function as the corresponding function 
in Pytorch, and the prefix Complex is added before the corresponding function. as the following modules.

If you have questions regarding the functionality or parameters of any function, please refer to the original
code comments for detailed explanations.

### layer

- ComplexLayers

  `ComplexConv2d` `ComplexConv1d` `ComplexConv3d` `ComplexFlatten`  `ComplexConvTransposed2d`  `ComplexLinear`

- ComplexDropout

  `ComplexDropout2D` `ComplexDropout` `ComplexDropoutRespectively`

- ComplexPooling

  `ComplexAvgPool1D` `ComplexAvgPool2D` `ComplexAvgPool3D` `ComplexPolarAvgPooling2D` `ComplexMaxPool2D`  `ComplexUnPooling2D`

- ComplexUpSampooling

  `ComplexUpSampling` `ComplexUpSamplingBilinear2d` `ComplexUpSamplingNearest2d`
  

### function

- ComplexBatchNorm

  `ComplexBatchNorm` `ComplexBatchNorm1d` `ComplexBatchNorm2d`

### activation

- ComplexActivations

  `complex_relu` `complex_elu` `complex_exponential` `complex_sigmoid` `complex_tanh` `complex_hard_sigmoid`
  `complex_leaky_relu` `complex_selu` `complex_softplus` `complex_softsign` `complex_softmax`
  `modrelu` `zrelu` `complex_cardioid` `sigmoid_real` `softmax_real_with_abs` `softmax_real_with_avg`
  `softmax_real_with_mult` `softmax_of_softmax_real_with_mult` `softmax_of_softmax_real_with_avg`
  `softmax_real_by_parameter` `softmax_real_with_polar` `georgiou_cdbp` `complex_signum`
  `mvn_activation` `apply_pol` `pol_tanh` `pol_sigmoid` `pol_selu`

### loss

- ComplexLoss

  `ComplexAverageCrossEntropy`  `ComplexAverageCrossEntropyAbs` `ComplexMeanSquareError` 
  `ComplexAverageCrossEntropyIgnoreUnlabeled`  `ComplexWeightedAverageCrossEntropy`
  `ComplexWeightedAverageCrossEntropyIgnoreUnlabeled`

---

## Code Examples

- MNIST Example
  in this example, complex values are not particularly useful, Takes the MINST dataset data as the real part and sets the imaginary part to 0 as input.
  The network structure of test2 is a simple MLP, and the network structure of test3 is the CVCNN network mentioned in the [article](https://github.com/wavefrontshaping/complexPyTorch).

The network structure is as follows.

```
# test2
class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fl = ComplexFlatten(start_dim=1)
        self.fc1 = ComplexLinear(784, 256)
        self.fc2 = ComplexLinear(256, 10)

    def forward(self, x):
        x = self.fl(x)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_softmax(x)
        return x
```

```
# test3
class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.maxpool1 = ComplexMaxPool2D(2, 2)
        self.bn2d = ComplexBatchNorm2d(10, track_running_stats=False)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.maxpool2 = ComplexMaxPool2D(2, 2)
        self.fc1 = ComplexLinear(4 * 4 * 20, 500)
        self.dropout = ComplexDropout2D(p=0.3)
        self.bn1d = ComplexBatchNorm1d(500, track_running_stats=False)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = self.maxpool1(x)
        x = self.bn2d(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, 4 * 4 * 20)
        x = self.fc1(x)
        x = self.dropout(x)
        x = complex_relu(x)
        x = self.bn1d(x)
        x = self.fc2(x)
        x = complex_softmax(x)
        return x
```

- Mstar Example
  The MSTAR data set is preprocessed, and the input data is converted to complex form and normalized.
  The network structure of test4 is same as test3，and the network structure of test5 is the A-ConvNets network mentioned in  S. Feng, K. Ji, F. Wang, L. Zhang, X. Ma and G. Kuang, "Electromagnetic Scattering Feature (ESF) Module Embedded Network Based on ASC Model for Robust and Interpretable SAR ATR," in  *IEEE Transactions on Geoscience and Remote Sensing* , vol. 60, pp. 1-15, 2022, Art no. 5235415, doi: 10.1109/TGRS.2022.3208333.

The network structure is as follows.

```
# test4
class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.maxpool1 = ComplexMaxPool2D(2, 2)
        self.bn2d = ComplexBatchNorm2d(10, track_running_stats=False)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.maxpool2 = ComplexMaxPool2D(2, 2)
        self.fc1 = ComplexLinear(47 * 47 * 20, 500)
        self.dropout = ComplexDropout2D(p=0.3)
        self.bn1d = ComplexBatchNorm1d(500, track_running_stats=False)
        self.fc2 = ComplexLinear(500, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = self.maxpool1(x)
        x = self.bn2d(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, 47 * 47 * 20)
        x = self.fc1(x)
        x = self.dropout(x)
        x = complex_relu(x)
        x = self.bn1d(x)
        x = self.fc2(x)
        x = complex_softmax(x)
        return x
```

```
class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 16, 13, 1)
        self.bn2d1 = ComplexBatchNorm2d(16, track_running_stats=False)
        self.maxpool1 = ComplexMaxPool2D(2, 2)
        self.conv2 = ComplexConv2d(16, 32, 13, 1)
        self.bn2d2 = ComplexBatchNorm2d(32, track_running_stats=False)
        self.maxpool2 = ComplexMaxPool2D(2, 2)
        self.conv3 = ComplexConv2d(32, 64, 12, 1)
        self.bn2d3 = ComplexBatchNorm2d(64, track_running_stats=False)
        self.maxpool3 = ComplexMaxPool2D(2, 2)
        self.dropout1 = ComplexDropout2D(p=0.5)
        self.conv4 = ComplexConv2d(64, 128, 10, 1)
        self.bn2d4 = ComplexBatchNorm2d(128, track_running_stats=False)
        self.conv5 = ComplexConv2d(128, 7, 6, 1)
        self.bn2d5 = ComplexBatchNorm2d(7, track_running_stats=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d1(x)
        x = complex_relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2d2(x)
        x = complex_relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn2d3(x)
        x = complex_relu(x)
        x = self.maxpool3(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.bn2d4(x)
        x = complex_relu(x)
        x = self.conv5(x)
        x = self.bn2d5(x)
        x = x.view(x.shape[0], -1)
        x = complex_softmax(x, 1)
        return x
```

---

## Acknowledgments

I would like to express my gratitude to my teacher Sirui Tian, my senior fellow apprentice Chen Chen, and all my peers in the research group for their guidance and assistance.