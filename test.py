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
parser.add_argument("--dataset_dir", default='./dataset', type=str, help="test_dataset dir")
parser.add_argument("--model", default='./log_centroid/DNAnet_evolution_error_max1_Navg_fromloss10_400.pth.tar', type=str, help="checkpoint")
parser.add_argument("--gpu", type=int, default=0, help="Test batch size")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--img_save_dir", type=str, default='./results_Navg/DNAnet_coarse/')#./results/
parser.add_argument("--dataset", type=list, default=['NUAA', 'NUDT', 'IRSTD-Ik','all'])

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)

def test(net, dataset):
    
    test_set = TestSetLoader_mask_dataset(opt.dataset_dir, dataset)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    eval_metric_IoU = mIoU(1) 
    eval_PD_FA = PD_FA(1,10)
    net.eval()
    for idx_iter, (img, gt_mask, gt_mask_centroid, size) in enumerate(test_loader):
        img = Variable(img).cuda()
        center_heatmap_pred = net.forward(img)
        if isinstance(center_heatmap_pred, list):
            center_heatmap_pred = center_heatmap_pred[-1]
            
        img_save = transforms.ToPILImage()((center_heatmap_pred[0,:,:size[0],:size[1]]).cpu())
        if not os.path.exists(opt.img_save_dir):
            os.makedirs(opt.img_save_dir)
        img_save.save(opt.img_save_dir + size[2][0] + '.png')
        
        eval_metric_IoU.update((center_heatmap_pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((center_heatmap_pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
        
    
    results1 = eval_metric_IoU.get_result()
    results2 = eval_PD_FA.get(idx_iter+1)
    print("Mean IOU with GT mask:\t" + str(results1))
    print("PD FA:\t" + str(results2))

def main():
    net = Net(diffusion=True).cuda()
    
    model = torch.load(opt.model)
    net.load_state_dict(model['state_dict'])
    for dataset in opt.dataset:
        test(net, dataset)

if __name__ == '__main__':
    main()