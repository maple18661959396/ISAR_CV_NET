import torch
from torch.nn import init
from torch.nn import Module, Parameter


class ComplexBatchNorm(Module):
    # Method described in the paper
    # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
    # BatchNorm to normalize the data for each batch into consistent distribution
    # This class simply defines how the relevant parameters are set for different inputs
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        """
        ComplexBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        num_features (int) – number of features or channels of the input
        eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional[float]) – the value used for the running_mean and running_var computation. Can be set to for 
        cumulative moving average (i.e. simple average). Default: 0.1None
        affine (bool) – a boolean value that when set to , this module has learnable affine parameters. Default: True True
        track_running_stats (bool) – a boolean value that when set to , this module tracks the running mean and variance, 
        and when set to , this module does not track such statistics, and initializes statistics buffers and as . 
        When these buffers are , this module always uses batch statistics. in both training and eval modes. 
        """
        if self.affine:
            #  True: learnable affine parameters
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:  # False: register
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_() # *.zero() Reassign to 0
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batcher_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)
            # Parameter reset


class ComplexBatchNorm1d(ComplexBatchNorm):
    # Applies Batch Normalization over a 2D or 3D input.
    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.complex64)
            mean_i = input.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0, unbiased=False) + self.eps
            Cii = input.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :] * input.real + Rri[None, :] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :] * input.imag + Rri[None, :] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0] * input.real + self.weight[None, :, 2] * input.imag + \
                     self.bias[None, :, 0]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2] * input.real + self.weight[None, :, 1] * input.imag + \
                            self.bias[None, :, 1]).type(torch.complex64)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input


class ComplexBatchNorm2d(ComplexBatchNorm):
    # Applies Batch Normalization over a 4D input.
    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batcher_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            # mean([0,2,3]) means averaging dimensions 0, 2, and 3 in turn， input is 4-dimensional data
            # batch_size * num_features * height * width
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
        input = input - mean[None, :, None, None]
        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim = [0, 2, 3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim = [0, 2, 3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3]) # .mul() 对应元素相乘
        else:
            n = input.numel() / input.size(1)
            Crr = self.running_covar[:, 0]+self.eps
            Cii = self.running_covar[:, 1]+self.eps
            Cri = self.running_covar[:, 2] # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None]*input.real+Rri[None, :, None, None]*input.imag).type(torch.complex64) \
                + 1j*(Rii[None, :, None, None]*input.imag+Rri[None, :, None, None]*input.real).type(torch.complex64)

        if self.affine:
            # True -> learnable parameters : weight and bias
            input = (self.weight[None, :, 0, None, None]*input.real+self.weight[None, :, 2, None, None]*input.imag + \
                     self.bias[None, :, 0, None, None]).type(torch.complex64) \
                     + 1j*(self.weight[None, :, 2, None, None]*input.real+self.weight[None, :, 1, None, None]*input.imag + \
                           self.bias[None, :, 1, None, None]).type(torch.complex64)

        return input