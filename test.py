import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DNAnet_evolution import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch LESPS")
parser.add_argument("--dataset_dir", default='../DNAnet/dataset/SIRST3', type=str, help="test_dataset dir")
parser.add_argument("--model", default='./log_centroid/DNAnet_evolution_error_max1_Navg_fromloss10_400.pth.tar', type=str, help="checkpoint")
parser.add_argument("--gpu", type=int, default=0, help="Test batch size")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--img_save_dir", type=str, default='./results_Navg/DNAnet_coarse/')#./results/
parser.add_argument("--dataset", type=list, default=['ACM', 'NUDT', 'IRSTD-Ik'])#'all', 

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)

def test(net, dataset):
    
    test_set = TestSetLoader_mask_dataset(opt.dataset_dir, dataset)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    eval_metric1 = mIoU(1) 
    eval_PD_FA = PD_FA1(1,10)
    # eval_metric2 = eval_metric_RJ(true_win=5, false_win=5)
    # eval_metric3 = eval_metric_RJ(true_win=5, false_win=5)
    net.eval()
    for idx_iter, (img, gt_mask, gt_mask_centroid, size) in enumerate(test_loader):
        img = Variable(img).cuda()
        # img = img.repeat(2,1,1,1)
        center_heatmap_pred = net.forward(img)#[-1]#[0:1,:,:,:]
        if isinstance(center_heatmap_pred, list):
            center_heatmap_pred = center_heatmap_pred[-1]
            
        img_save = transforms.ToPILImage()((center_heatmap_pred[0,:,:size[0],:size[1]]).cpu())
        if not os.path.exists(opt.img_save_dir):
            os.makedirs(opt.img_save_dir)
        img_save.save(opt.img_save_dir + size[2][0] + '.png')
        # opt.threshold = center_heatmap_pred.max()*0.5
        # map1 = center_heatmap_pred*(center_heatmap_pred>0.1)
        # opt.threshold = map1.sum()/len(torch.nonzero(map1))
        
        eval_metric1.update((center_heatmap_pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((center_heatmap_pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
        
        # img_centroid = torch.zeros(gt_mask.shape[-2], gt_mask.shape[-1])
        # img_max = torch.zeros(gt_mask.shape[-2], gt_mask.shape[-1])
        # label_image = measure.label((center_heatmap_pred>opt.threshold).squeeze().cpu())
        # for region in measure.regionprops(label_image, cache=False):
        #     #one centroid
        #     img_centroid[int(region.centroid[0]),int(region.centroid[1])] = 1
            
        #     # max points
        #     # point nbr
        #     point_nbr_mask = torch.zeros(gt_mask.shape[-2], gt_mask.shape[-1])
        #     point_nbr_mask[region.bbox[0]:region.bbox[2]+1, region.bbox[1]:region.bbox[3]+1] = 1
        #     point_nbr = center_heatmap_pred.cpu()*point_nbr_mask
        #     max_index = torch.nonzero(point_nbr==point_nbr.max())
        #     for index in max_index:
        #         img_max[index[-2], index[-1]] = 1
            
            
        # eval_metric2.update(gt_mask_centroid, img_centroid)    
        # eval_metric3.update(gt_mask_centroid, img_max)  
        
    
    results1 = eval_metric1.get_result()
    results2 = eval_PD_FA.get(idx_iter+1)
    # results2 = eval_metric2.get_results()
    # results3 = eval_metric3.get_results()
    print("Mean IOU with GT mask:\t" + str(results1))
    print("PD FA:\t" + str(results2))
    # print("RJ centroid results:\t" + str(results2))
    # print("RJ max results:\t" + str(results3))

def main():
    net = Net(diffusion=False).cuda()
    # log_list = ['./log_mask/']#'./log_coarse/','./log_rand/','./log_centroid/',
    # for log_dir in log_list:
    #     model = torch.load(log_dir + opt.model)
    #     net.load_state_dict(model['state_dict'])
    #     for dataset in opt.dataset:
    #         test(net, dataset)
    model = torch.load(opt.model)
    net.load_state_dict(model['state_dict'])
    for dataset in opt.dataset:
        test(net, dataset)

if __name__ == '__main__':
    main()