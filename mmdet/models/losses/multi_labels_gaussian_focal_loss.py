# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

def compute_weights(gaussian_target, fill_value):
    weights = torch.full_like(gaussian_target, fill_value, dtype=torch.bool)
    nonzero_indices = (gaussian_target == 1).nonzero(as_tuple=True)
      
    center_x, center_y = nonzero_indices
    for x, y in zip(center_x, center_y):
        weights[max(0, x-1):min(gaussian_target.shape[0], x+2), 
                max(0, y-1):min(gaussian_target.shape[1], y+2)] = not fill_value
    return weights

def compute_pos_weights(gaussian_target):
    return compute_weights(gaussian_target, False)  

def compute_neg_weights(gaussian_target):
    return compute_weights(gaussian_target, True) 
 


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def multi_labels_gaussian_focal_loss(pred, gaussian_target, alpha=3.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): Defaults to 3.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12

    pos_weights = compute_pos_weights(gaussian_target)
    neg_weights = compute_neg_weights(gaussian_target)

    pos_loss = -(1-abs(gaussian_target-pred)).log() * (abs(gaussian_target - pred)).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights * (1 - gaussian_target).pow(gamma)
    return pos_loss + neg_loss


@LOSSES.register_module()
class MultiLabelsGaussianFocalLoss(nn.Module):
    """MultiLabelsGaussianFocalLoss is a variant of GaussianFocalLoss.


    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(MultiLabelsGaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * multi_labels_gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
