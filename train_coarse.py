import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DNAnet_evolution import  Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from skimage import morphology
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch LESPS")
parser.add_argument("--save_perdix", default='DNAnet_evolution', type=str, help="Save path")#gaussian_diffusion_DNAnet_deep_drop_500
parser.add_argument("--save", default='./log_coarse', type=str, help="Save path")
parser.add_argument("--resume", default='', type=str, help="Resume path (default: none)")#./log_show/DANnet_centroid_50.pth.tar
parser.add_argument("--dataset_dir", default='./ataset/SIRST3', type=str, help="train_dataset")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs to train for")
parser.add_argument("--gpu", default=0, type=int, help="gpu ids (default: 0)")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument("--step", type=int, default=[200, 300], help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)

with open(opt.dataset_dir+'/50_50/train.txt', 'r') as f:
    train_list = f.read().splitlines()
opt.train_mask_list = []
for idx in range(len(train_list)):
    mask = Image.open(opt.dataset_dir + '/masks_coarse/' + train_list[idx] + '.png')
    mask = np.array(mask, dtype=np.float32)  / 255.0
    opt.train_mask_list.append(mask)
opt.train_mask_list = None  

opt.f = open('./DNAnet_evolution_coarse.txt', 'w')

train_iou_list = []
test_iou_list = []

def train(train_loader, epoch_num):
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    update_epoch_loss = []
    start_loss = 10
    start_click = 0
    end_click = 0
    
    
    net = Net().cuda()
    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        for i in range(len(opt.step)):
            opt.step[i] = opt.step[i] - epoch_state

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.step, gamma=opt.gamma)
    
    for idx_epoch in range(epoch_state, epoch_num):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            center_heatmap_pred = net.forward(img)
            loss = net.loss(center_heatmap_pred, gt_mask)
            # update(net)
            total_loss_epoch.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write('\n')
            total_loss_epoch = []
        
        # end update?
        if end_click == 0:
            # subsequent update
            if start_click == 1 and (idx_epoch + 1) % 5 == 0:#total_loss_list[-1] <  update_epoch_loss[-1]:
                loss_before, loss_after = update_gt_mask(net, thresh=0.5)
                save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_path=opt.save, filename= opt.save_perdix + '_' + str(idx_epoch + 1) + '.pth.tar')
                test(net)
                update_epoch_loss.append(total_loss_list[-1])
            
            # first update    
            if idx_epoch > 0 and total_loss_list[-1] < start_loss and start_click == 0:
                print('update start')
                start_click = 1
                loss_before, loss_after = update_gt_mask(net, thresh=0.5)
                test(net)
                update_epoch_loss.append(total_loss_list[-1])
                
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                    'train_iou_list': train_iou_list,
                    'test_iou_list': test_iou_list,
                    }, save_path=opt.save, filename= opt.save_perdix + '_' + str(idx_epoch + 1) + '.pth.tar')
            
        else: 
            if (idx_epoch + 1) % 10 == 0:
                print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
                test(net)

def update_gt_mask(net, thresh=0.7, is_initial=False):
    test_set = Update_mask_coarse(opt.dataset_dir, update_dir=opt.save_perdix, mask_update_list=opt.train_mask_list)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    eval_metric1 = mIoU(1) 
    net.eval()
    loss_before = []
    loss_after = []
    for idx_iter, (img, gt_mask, gt_mask_centroid, update_dir, size) in tqdm(enumerate(test_loader)):
        img, gt_mask_centroid = Variable(img).cuda(), Variable(gt_mask_centroid).cuda()
        center_heatmap_pred = net.forward(img)
        loss_before.append((net.loss(center_heatmap_pred, gt_mask_centroid)).detach().cpu())
        if is_initial == True:
            update_mask = net.update_gt_intial(gt_mask_centroid, center_heatmap_pred, thresh, size, initial=is_initial)
        else:
            update_mask = net.update_gt(gt_mask_centroid, center_heatmap_pred, thresh, size, initial=is_initial)
        if isinstance(update_dir, torch.Tensor):
            opt.train_mask_list[update_dir] = update_mask[0,0,:size[0],:size[1]].cpu().detach().numpy()
        else:
            img_save = transforms.ToPILImage()((update_mask[0,:,:size[0],:size[1]]).cpu())
            img_save.save(update_dir[0])
        loss_after.append((net.loss(center_heatmap_pred, update_mask)).detach().cpu())
        eval_metric1.update((update_mask>=0.5).cpu(), gt_mask)
        
    results1 = eval_metric1.get_result()
    opt.f.write("Mean IOU with GT mask:\t" + str(results1))
    opt.f.write('\n')
    opt.f.write(time.ctime()[4:-5] + ' loss_before---%f, loss_after---%f,' 
                  % (float(np.array(loss_before).mean()), float(np.array(loss_after).mean())))
    opt.f.write('\n')
    print("Mean IOU with GT mask:\t" + str(results1))
    print(time.ctime()[4:-5] + ' loss_before---%f, loss_after---%f,' 
                  % (float(np.array(loss_before).mean()), float(np.array(loss_after).mean())))
    train_iou_list.append(results1[1])
    return loss_before, loss_after

def test(net):
    test_set = TestSetLoader_mask(opt.dataset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    eval_metric1 = mIoU(1) 
    net.eval()
    for idx_iter, (img, gt_mask, gt_mask_centroid) in enumerate(test_loader):
        img = Variable(img).cuda()
        center_heatmap_pred = net.forward(img)
        if isinstance(center_heatmap_pred, list):
            center_heatmap_pred = center_heatmap_pred[-1]
        eval_metric1.update((center_heatmap_pred>0.5).cpu(), gt_mask)
        
    
    results1 = eval_metric1.get_result()
    # results3 = eval_metric3.get_results()
    opt.f.write("Mean IOU with GT mask:\t" + str(results1))
    opt.f.write('\n')
    print("Mean IOU with GT mask:\t" + str(results1))
    test_iou_list.append(results1[1])


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def main():
    train_set = TrainSetLoader_coarse(opt.dataset_dir, patch_size=opt.patchSize, update_dir=opt.save_perdix, mask_update_list=opt.train_mask_list)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    train(train_loader, opt.nEpochs)

if __name__ == '__main__':
    main()
    opt.f.close()