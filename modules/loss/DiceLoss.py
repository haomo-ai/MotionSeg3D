#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice

# https://smp.readthedocs.io/en/latest/losses.html
# https://github.com/pytorch/pytorch/issues/1249
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        # input_soft = F.softmax(input, dim=1) # have done is network last layer

        # create the labels one hot tensor
        # target_one_hot = one_hot(target, num_classes=input.shape[1],
        #                          device=input.device, dtype=input.dtype)
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2)

        # compute the actual dice score
        dims = (1, 2, 3)
        # intersection = torch.sum(input_soft * target_one_hot, dims)
        # cardinality = torch.sum(input_soft + target_one_hot, dims)

        ## if we need to ignore the class=0
        input_filter = input[:, 1:, :, :]
        target_one_hot_filter = input[:, 1:, :, :]
        intersection = torch.sum(input_filter * target_one_hot_filter, dims)
        cardinality = torch.sum(input_filter + target_one_hot_filter, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)



######################
# functional interface
######################


def dice_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)

