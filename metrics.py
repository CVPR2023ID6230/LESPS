import os
import numpy as np
import scipy.io as sio
import xml.dom.minidom as doxml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import measure
import  numpy

class eval_metric():
    def __init__(self, th, conf_th):
        super(eval_metric, self).__init__()
        self.th = th
        self.conf_th = conf_th

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.metric = self.dist_batch

    def update(self, gt, det):
        if det.shape[0] > 0 and det.shape[1] > 2:
            det = det[det[:,2] > self.conf_th]
            det = det[:,:2]
        if gt.shape[0] > 0:
            if det.shape[0] > 0:
                #get metric
                matrix = self.metric(det, gt)
                ##linear assignment
                # dist_matrix[dist_matrix > dis_th] = dis_th + 10
                # dist_matrix[dist_matrix > dis_th] = np.inf
                dist_matrix1 = matrix / (matrix.max() + 1e-2)
                matrix[matrix > self.th] = (dist_matrix1[matrix > self.th] + 1) * self.th
                matched_indices = self.linear_assignment(matrix)
                matched_iou = matrix[matched_indices[:, 0], matched_indices[:, 1]]
                matched_iou = matched_iou[matched_iou < self.th]
                #compute the results
                tp = matched_iou.shape[0]
                fn = gt.shape[0] - tp
                fp = det.shape[0] - tp
            else:
                tp = 0
                fn = gt.shape[0]
                fp = 0
        else:
            tp = 0
            fn = 0
            fp = det.shape[0]

        self.tp += tp
        self.fn += fn
        self.fp += fp

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def get_result(self):
        prec = self.tp/(self.tp+self.fp+1e-7)
        recall = self.tp/(self.tp+self.fn+1e-7)
        f1 = 2*recall*prec/(recall+prec+1e-7)

        out = {}
        out['recall'] = recall
        out['prec'] = prec
        out['f1'] = f1
        out['tp'] = self.tp
        out['fp'] = self.fp
        out['fn'] = self.fn
        return out


    def dist_batch(self, det_center, gt_center):
        gt_center = np.expand_dims(gt_center, 0)
        det_center = np.expand_dims(det_center, 1)
        o = np.sqrt(np.sum((gt_center - det_center) ** 2, -1))
        return (o)

    def linear_assignment(self, cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])  #
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))
        
class mIoU():
    
    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get_result(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class eval_metric_RJ():
    def __init__(self, true_win=3, false_win=5):
        super(eval_metric_RJ, self).__init__()
        self.true_win = true_win
        self.false_win = false_win
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, gt_mask, det_mask):
        tp = 0
        for (pre_h, pre_w) in torch.nonzero(gt_mask)[:,2:4]:
            TP_point_mask = gt_mask.new_zeros(gt_mask.shape)
            TP_point_mask[0, 0, pre_h, pre_w] = 1
            TP_point_mask = F.conv2d(TP_point_mask, weight=torch.ones(1,1,self.true_win,self.true_win), stride=1, padding=self.true_win//2)
            tp = tp + ((TP_point_mask * det_mask).sum() > 0).float()
            
        # gt_mask_TP = F.conv2d(gt_mask, weight=torch.ones(1,1,3,3), stride=1, padding=1)
        gt_mask_missing = 1-F.conv2d(gt_mask, weight=torch.ones(1,1,self.false_win,self.false_win), stride=1, padding=self.false_win//2)
        # tp = ((gt_mask_TP * det_mask).sum() > 0).float()
        fp = (gt_mask_missing * det_mask).sum()
        fn = gt_mask.sum() - tp
        
        self.tp += tp
        self.fn += fn
        self.fp += fp

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def get_results(self):
        prec = self.tp/(self.tp+self.fp+1e-7)
        recall = self.tp/(self.tp+self.fn+1e-7)
        f1 = 2*recall*prec/(recall+prec+1e-7)

        out = {}
        out['recall'] = recall
        out['prec'] = prec
        out['f1'] = f1
        out['tp'] = self.tp
        out['fp'] = self.fp
        out['fn'] = self.fn
        return out

def batch_pix_accuracy(output, target):
    
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union



class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA[0],Final_PD[0]


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])


class PD_FA1():
    def __init__(self, nclass, bins):
        super(PD_FA1, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        predits  = np.array((preds > 0).cpu()).astype('int64')
        # predits  = np.reshape (predits,  (256,256))
        labelss = np.array((labels).cpu()).astype('int64') # P
        # labelss = np.reshape (labelss , (256,256))

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel +=np.sum(self.dismatch)
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target

        return float(Final_FA.cpu().detach().numpy()), Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    
    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos