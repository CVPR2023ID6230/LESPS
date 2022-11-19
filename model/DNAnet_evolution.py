from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import *
from skimage import measure
import os
from utils.gaussian_target import *
from model.DANnet.model_DNANet import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        
        self.backbone = nn.Sequential(
            DNANet(num_classes=1,input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True)  
        )
        
        self.loss_center_heatmap = GaussianFocalLoss(loss_weight=1.0)#, protect=True)
        
    def forward(self, img):
        center_heatmap_pred = self.backbone(img)
        return center_heatmap_pred

    def loss(self,
             center_heatmap_pred_list,
             gt_mask):
        """Compute losses of the head.

        Args:
            center_heatmap_pred (Tensor): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (Tensor): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (Tensor): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_mask (Tensor): Ground truth masks for each image with
                shape (B, 1, H, W).

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_offset (Tensor): loss of offset heatmap.
        """

        center_heatmap_target, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
        loss_center_heatmap = 0
        for center_heatmap_pred in center_heatmap_pred_list:
            loss_center_heatmap = loss_center_heatmap + self.loss_center_heatmap(
                center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        return loss_center_heatmap
        
    def update_gt(self, gt_masks, center_heatmap_pred, thresh, size):
        center_heatmap_pred = center_heatmap_pred[-1]
        bs, c, feat_h, feat_w = center_heatmap_pred.shape
        update_gt_masks = gt_masks.clone()
        background_length = 33
        target_length = 3
        
        label_image = measure.label((gt_masks[0,0,:,:]>0.5).cpu())
        for region in measure.regionprops(label_image, cache=False):
            cur_point_mask = center_heatmap_pred.new_zeros(bs, c, feat_h, feat_w)
            cur_point_mask[0, 0, int(region.centroid[0]), int(region.centroid[1])] = 1
            nbr_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,background_length,background_length).to(gt_masks.device), stride=1, padding=background_length//2))>0).float()
            targets_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,target_length,target_length).to(gt_masks.device), stride=1, padding=target_length//2))>0).float()
            max_limitation = size[0] * size[1] * 0.0015 * 2
            threshold_start = (center_heatmap_pred * nbr_mask ).max()*thresh
            threshold_delta = ((center_heatmap_pred * nbr_mask ).max() - threshold_start) * (len(region.coords)/max_limitation).to(gt_masks.device)
            threshold = threshold_start + threshold_delta
            thresh_mask = (center_heatmap_pred * nbr_mask > threshold).float()
            
            label_image = measure.label((thresh_mask[0,:,:,:]>0).cpu())
            if label_image.max() > 1:
                for num_cur in range(label_image.max()):
                    curr_mask = thresh_mask * torch.tensor(label_image == (num_cur + 1)).float().unsqueeze(0).to(gt_masks.device) #  torch.tensor(）
                    if (curr_mask * targets_mask).sum() == 0:
                        thresh_mask = thresh_mask - curr_mask
            
            pred_mask_nbr = (center_heatmap_pred * thresh_mask)
            target_patch = (update_gt_masks * thresh_mask + pred_mask_nbr)/2
            background_patch = update_gt_masks * (1-thresh_mask)
            
            update_gt_masks = background_patch + target_patch
            
            
        update_gt_masks = torch.max(update_gt_masks, (gt_masks==1).float())

        return update_gt_masks
