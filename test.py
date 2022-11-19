import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.dataset import *
import matplotlib.pyplot as plt
from utils.metrics import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch LESPS")
parser.add_argument("--dataset_dir", default='./dataset', type=str, help="test_dataset dir")
parser.add_argument("--model", default='ALCnet_centroid', type=str, help="checkpoint")
parser.add_argument("--gpu", type=int, default=0, help="Test batch size")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--img_save_dir", type=str, default='./results/')
parser.add_argument("--dataset", type=list, default=['NUAA', 'NUDT', 'IRSTD-Ik'])#,'all'

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
        
        img_save_dir = opt.img_save_dir + opt.model + '/'
        img_save = transforms.ToPILImage()((center_heatmap_pred[0,:,:size[0],:size[1]]).cpu())
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        img_save.save(img_save_dir + size[2][0] + '.png')
        
        eval_metric_IoU.update((center_heatmap_pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((center_heatmap_pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
        
    
    results1 = eval_metric_IoU.get_result()
    results2 = eval_PD_FA.get(idx_iter+1)
    print("Mean IOU with GT mask:\t" + str(results1))
    print("PD FA:\t" + str(results2))

def main():
    model_name, label_name = opt.model.split('_')
    
    if model_name == 'DNAnet':
        from model.DNAnet_evolution import Net
    elif model_name == 'ACM':
        from model.ACM_evolution import Net
    elif model_name == 'ALCnet':
        from model.ALCnet_evolution import Net
    
    ckpt = './log/' + opt.model + '.pth.tar'
    
    net = Net().cuda()
    
    model = torch.load(ckpt)
    net.load_state_dict(model['state_dict'])
    for dataset in opt.dataset:
        test(net, dataset)

if __name__ == '__main__':
    main()