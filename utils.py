import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def nms_point(point_array, threshold=3):
    x1 = point_array[:, 0]
    y1 = point_array[:, 1]
    scores = point_array[:, 2]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = x1[i] - x1[order[1:]]
        yy1 = y1[i] - y1[order[1:]]
        distance = np.sqrt(xx1 * xx1 + yy1 * yy1)
        inds = np.where(distance > threshold)[0]
        order = order[inds + 1]
    point_array_out = point_array[keep]
    return point_array_out
        
        
def random_crop(img, mask, patch_size): # HR: N*H*W
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size


    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch

def add_noise(img, n_std):
    return img + np.random.normal(0, n_std, img.shape)

def add_light(img, light, *paras, mode):
    if mode == 'point':
        x0, y0, radius = paras
        light_res = np.zeros(3, radius, radius)
        for i in range(radius):
            for j in range(radius):
                light_res[0, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[1, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[2, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)

        light_res = np.clip(light_res + img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :], 0, 255)
        img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :] = light_res
    return img

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
        
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

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
                 loss_weight=1.0,
                 protect=False):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.protect = protect

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
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            protect=self.protect)
        loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
        return loss_reg

class GaussianFocalLoss_o(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

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
        super(GaussianFocalLoss_o, self).__init__()
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
        loss_reg = self.loss_weight * gaussian_focal_loss_o(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma)
        loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
        return loss_reg
   
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
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
            target (torch.Tensor): The learning target of the prediction.
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
        loss_bbox = self.loss_weight * l1_loss(
            pred, target)
        loss_bbox = weight_reduce_loss(loss_bbox, weight, reduction, avg_factor)
        return loss_bbox

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0, protect=False):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target#.eq(1)
    if protect:
        gaussian_target = (F.conv2d(gaussian_target, weight=torch.ones(1,1,33,33).to(gaussian_target.device), stride=1, padding=33//2)>0).float()
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

def gaussian_focal_loss_o(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss