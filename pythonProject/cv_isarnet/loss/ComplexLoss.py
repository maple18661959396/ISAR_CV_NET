import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexAverageCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self):
        super(ComplexAverageCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        real_loss = F.cross_entropy(y_pred.real, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        if y_pred.dtype.is_complex:
            imag_loss = F.cross_entropy(y_pred.imag, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2.
# ComplexAverageCrossEntropy Use the torch.nn.functional.cross_entropy in the real and imaginary parts, respectively.
# Then take average


class ComplexAverageCrossEntropyAbs(nn.CrossEntropyLoss):
    def __init__(self):
        super(ComplexAverageCrossEntropyAbs, self).__init__()

    def forward(self, y_pred, y_true):
        if y_pred.dtype.is_complex:
            loss = F.cross_entropy(torch.abs(y_pred), y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            loss = F.cross_entropy(y_pred, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss
# Input complex tensor take modulo values then Use the torch.nn.functional.cross_entropy.


class ComplexMeanSquareError(nn.MSELoss):
    def __init__(self):
        super(ComplexMeanSquareError, self).__init__()

    def forward(self, y_pred, y_true):
        if y_pred.dtype.is_complex and not y_true.dtype.is_complex:
            y_true = y_true.type(torch.complex64) + 1j * y_true.type(torch.complex64)
        y_true = y_true.type(y_pred.dtype)
        return torch.mean(torch.square(torch.abs(y_true - y_pred)))
# return F.mse_loss(torch.abs(y_true), torch.abs(y_pred), reduction=self.reduction)
# The absolute value of the difference, not the difference of the absolute value.


class ComplexAverageCrossEntropyIgnoreUnlabeled(ComplexAverageCrossEntropy):
    # Remove the unlabeled part of the corresponding sample
    def forward(self, y_pred, y_true):
        mask = torch.any(y_true.detype(bool), -1)
        # Tests if any element in evaluates to True.
        y_true = torch.masked_select(y_true, mask)
        # Return a new 1-D tensor which indexes the input tensor according to the boolean mask which is a BoolTensor.
        y_pred = torch.masked_select(y_pred, mask)
        return super(ComplexAverageCrossEntropyIgnoreUnlabeled, self).forward(y_pred, y_true)


class ComplexWeightedAverageCrossEntropy(ComplexAverageCrossEntropy):
    # The weight is updated by the given weight and the true value
    def __init__(self, weights, **kwargs):
        self.class_weights = weights
        super(ComplexWeightedAverageCrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
        weights = torch.sum(self.class_weights * y_true, -1)
        unweighted_losses = super(ComplexWeightedAverageCrossEntropy, self).forward(y_pred, y_true)
        # unweighted loss function
        weighted_losses = unweighted_losses * weights.dtype(unweighted_losses.dtype)
        # Add weight
        return weighted_losses


class ComplexWeightedAverageCrossEntropyIgnoreUnlabeled(ComplexAverageCrossEntropy):
    # unlabeled part is removed then weighted according to the corresponding weight
    def __init__(self, weights, **kwargs):
        self.class_weights = weights
        super(ComplexWeightedAverageCrossEntropyIgnoreUnlabeled, self).__init__()

    def forward(self, y_pred, y_true):
        mask = torch.any(y_true.dtype(bool), -1)
        y_true = torch.masked_select(y_true, mask)
        y_pred = torch.masked_select(y_pred, mask)
        # Weighted the processed data
        weights = torch.sum(self.class_weights * y_true, -1)
        unweighted_losses = super(ComplexWeightedAverageCrossEntropyIgnoreUnlabeled, self).forward(y_pred, y_true)
        # unweighted loss function
        weighted_losses = unweighted_losses * weights.dtype(unweighted_losses.dtype)
        # Add weight
        return weighted_losses



class ComplexL1Loss(nn.L1Loss):
    def __init__(self, reduction='mean'):
        super(ComplexL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true, reduction=self.reduction)
# F.l1_loss supports real-valued and complex-valued inputs.


class ComplexSmoothL1Loss(nn.L1Loss):
    def __init__(self, beta: float = 1.0):
        super(ComplexSmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        return F.smooth_l1_loss(y_pred, y_true, reduction=self.reduction, beta=self.beta)
# F.smooth_l1_loss supports real-valued and complex-valued inputs.


class ComplexNLLLoss(nn.NLLLoss):
    def __init__(self):
        super(ComplexNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        real_loss = F.nll_loss(y_pred.real, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        if y_pred.dtype.is_complex:
            imag_loss = F.nll_loss(y_pred.imag, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2.
# ComplexNLLLoss Use the torch.nn.functional.nll_loss in the real and imaginary parts, respectively. Then take average


class ComplexNLLLossAbs(nn.NLLLoss):
    def __init__(self):
        super(ComplexNLLLossAbs, self).__init__()

    def forward(self, y_pred, y_true):
        if y_pred.dtype.is_complex:
            loss = F.nll_loss(torch.abs(y_pred), y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            loss = F.nll_loss(y_pred, y_true, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss
# Input complex tensor take modulo values then Use the torch.nn.functional.nll_loss.
"""
y_pred = torch.randn(3, 3) + 1j * 2 * torch.rand(3, 3)
y_true = torch.randn(3, 3)
loss = ComplexMeanSquareError().forward(y_true, y_pred)
print(loss)
"""