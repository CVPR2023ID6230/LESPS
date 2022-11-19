from utils import *
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TestSetLoader_mask(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader_mask).__init__()
        self.dataset_dir = dataset_dir
        with open(dataset_dir+'/50_50/test.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        mask_centroid = Image.open(self.dataset_dir + '/masks_centroid/' + self.train_list[idx] + '.png')
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        mask_centroid = np.array(mask_centroid, dtype=np.float32)  / 255.0
        
        h, w = img.shape
        img = np.pad(img, ((0, (h//32+1)*32-h),(0, (w//32+1)*32-w)), mode='constant')
        mask = np.pad(mask, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        mask_centroid = np.pad(mask_centroid, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        
        img, mask, mask_centroid = img[np.newaxis,:], mask[np.newaxis,:], mask_centroid[np.newaxis,:]
        

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_centroid = torch.from_numpy(np.ascontiguousarray(mask_centroid))
        return img, mask, mask_centroid
    def __len__(self):
        return len(self.train_list) 

class Update_mask(Dataset):
    def __init__(self, dataset_dir, update_dir='', mask_update_list=None):
        super(Update_mask).__init__()
        self.update_dir = update_dir
        self.dataset_dir = dataset_dir
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        self.mask_update_list = mask_update_list
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        if self.mask_update_list == None:
            mask_centroid = Image.open(self.dataset_dir + '/masks_centroid_update_' + self.update_dir + '/' + self.train_list[idx] + '.png')
            update_dir = self.dataset_dir + '/masks_centroid_update_' + self.update_dir + '/' + self.train_list[idx] + '.png'
            mask_centroid = np.array(mask_centroid, dtype=np.float32)  / 255.0
        else:
            mask_centroid = self.mask_update_list[idx]
            update_dir = idx
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        # mask_centroid = np.array(mask_centroid, dtype=np.float32)  / 255.0
        
        h, w = img.shape
        img = np.pad(img, ((0, (h//32+1)*32-h),(0, (w//32+1)*32-w)), mode='constant')
        mask = np.pad(mask, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        mask_centroid = np.pad(mask_centroid, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        
        img, mask, mask_centroid = img[np.newaxis,:], mask[np.newaxis,:], mask_centroid[np.newaxis,:]
        

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_centroid = torch.from_numpy(np.ascontiguousarray(mask_centroid))
        return img, mask, mask_centroid, update_dir, [h,w]
    def __len__(self):
        return len(self.train_list) 

class Update_mask_coarse(Dataset):
    def __init__(self, dataset_dir, update_dir='', mask_update_list=None):
        super(Update_mask_coarse).__init__()
        self.update_dir = update_dir
        self.dataset_dir = dataset_dir
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        self.mask_update_list = mask_update_list
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        if self.mask_update_list == None:
            mask_centroid = Image.open(self.dataset_dir + '/masks_coarse_update_' + self.update_dir + '/' + self.train_list[idx] + '.png')
            update_dir = self.dataset_dir + '/masks_coarse_update_' + self.update_dir + '/' + self.train_list[idx] + '.png'
            mask_centroid = np.array(mask_centroid, dtype=np.float32)  / 255.0
        else:
            mask_centroid = self.mask_update_list[idx]
            update_dir = idx
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        
        
        h, w = img.shape
        img = np.pad(img, ((0, (h//32+1)*32-h),(0, (w//32+1)*32-w)), mode='constant')
        mask = np.pad(mask, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        mask_centroid = np.pad(mask_centroid, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        
        img, mask, mask_centroid = img[np.newaxis,:], mask[np.newaxis,:], mask_centroid[np.newaxis,:]
        

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_centroid = torch.from_numpy(np.ascontiguousarray(mask_centroid))
        return img, mask, mask_centroid, update_dir, [h,w]
    def __len__(self):
        return len(self.train_list) 



class TestSetLoader_mask_dataset(Dataset):
    def __init__(self, dataset_dir, dataset):
        super(TestSetLoader_mask_dataset).__init__()
        self.dataset_dir = dataset_dir
        if dataset == 'all':
            with open(dataset_dir+'/50_50/test.txt', 'r') as f:
                self.train_list = f.read().splitlines()
        else:
            with open(dataset_dir+'/50_50/test_' + dataset + '.txt', 'r') as f:
                self.train_list = f.read().splitlines()    
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        mask_centroid = Image.open(self.dataset_dir + '/masks_centroid/' + self.train_list[idx] + '.png')
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        mask_centroid = np.array(mask_centroid, dtype=np.float32)  / 255.0
        
        h, w = img.shape
        img = np.pad(img, ((0, (h//32+1)*32-h),(0, (w//32+1)*32-w)), mode='constant')
        mask = np.pad(mask, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        mask_centroid = np.pad(mask_centroid, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        
        img, mask, mask_centroid = img[np.newaxis,:], mask[np.newaxis,:], mask_centroid[np.newaxis,:]
        

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_centroid = torch.from_numpy(np.ascontiguousarray(mask_centroid))
        return img, mask, mask_centroid, [h,w, self.train_list[idx]]
    def __len__(self):
        return len(self.train_list) 

import shutil
class TrainSetLoader_centroid(Dataset):
    def __init__(self, dataset_dir=None, patch_size=None, update_dir='', update_mode=False, mask_update_list=None):
        super(TrainSetLoader_centroid).__init__()
        if mask_update_list == None:
            self.update_dir = update_dir
            if update_dir != '' and update_mode==False:
                if os.path.exists(dataset_dir + '/masks_centroid_update_' + self.update_dir + '/'):
                    shutil.rmtree(dataset_dir + '/masks_centroid_update_' + self.update_dir + '/')
                shutil.copytree(dataset_dir + '/masks_centroid/', dataset_dir + '/masks_centroid_update_' + self.update_dir + '/')

        self.mask_update_list = mask_update_list   
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        if self.mask_update_list == None:
            if self.update_dir != '':
                mask = Image.open(self.dataset_dir + '/masks_centroid_update_' + self.update_dir + '/' + self.train_list[idx] + '.png')
            else:
                mask = Image.open(self.dataset_dir + '/masks_centroid/' + self.train_list[idx] + '.png')
            mask = np.array(mask, dtype=np.float32)  / 255.0
        else:
            mask = self.mask_update_list[idx]    
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]

        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TrainSetLoader_mask(Dataset):
    def __init__(self, dataset_dir, patch_size):
        super(TrainSetLoader_mask).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        
        # ### masks
        # img_mask = np.zeros((mask.shape[-2], mask.shape[-1]))
        # label_image = measure.label(mask)
        # for region in measure.regionprops(label_image, cache=False):
            # #centroid
            # delta_noise = 2
            # h_final = int(region.centroid[0] + delta_noise - (np.random.rand())* delta_noise*2)
            # w_final = int(region.centroid[1] + delta_noise - (np.random.rand())* delta_noise*2)
            # h_final = max(0,h_final)
            # h_final = min(h_final, mask.shape[-2]-1)
            # w_final = max(0,w_final)
            # w_final = min(w_final, mask.shape[-1]-1)
            # img_mask[h_final, w_final] = 1
            
            # #max_max
            # axis_0 = region.coords[region.coords[:,0]==region.coords[:,0].max()]
            # axis_01 = axis_0[axis_0[:,1]==axis_0[:,1].max()]
            # img_mask[axis_01[0][0],axis_01[0][1]] = 1
            #max_min
            # axis_0 = region.coords[region.coords[:,0]==region.coords[:,0].max()]
            # axis_01 = axis_0[axis_0[:,1]==axis_0[:,1].min()]
            # img_mask[axis_01[0][0],axis_01[0][1]] = 1
            # #min_max
            # axis_0 = region.coords[region.coords[:,0]==region.coords[:,0].min()]
            # axis_01 = axis_0[axis_0[:,1]==axis_0[:,1].max()]
            # img_mask[axis_01[0][0],axis_01[0][1]] = 1
            # #min_min
            # axis_0 = region.coords[region.coords[:,0]==region.coords[:,0].min()]
            # axis_01 = axis_0[axis_0[:,1]==axis_0[:,1].min()]
            # img_mask[axis_01[0][0],axis_01[0][1]] = 1
            
        img_mask = mask
        img_patch, mask_patch = random_crop(img, img_mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]

        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TrainSetLoader_rand(Dataset):
    def __init__(self, dataset_dir, patch_size, update_dir='', update_mode=False, mask_update_list=None):
        super(TrainSetLoader_rand).__init__()
        self.dataset_dir = dataset_dir
        if mask_update_list == None:
            self.update_dir = update_dir
            if self.update_dir != '' and update_mode==False:
                # if not os.path.exists(dataset_dir + '/masks_centroid_update/'):
                #     os.makedirs(dataset_dir + '/masks_centroid_update/')
                if os.path.exists(dataset_dir + '/masks_rand_update_' + self.update_dir + '/'):
                    shutil.rmtree(dataset_dir + '/masks_rand_update_' + self.update_dir + '/')
                shutil.copytree(dataset_dir + '/masks_rand/', dataset_dir + '/masks_rand_update_' + self.update_dir + '/')
        else:
            self.mask_update_list = mask_update_list
        self.patch_size = patch_size
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        if self.mask_update_list == None:
            if self.update_dir != '':
                mask = Image.open(self.dataset_dir + '/masks_rand_update_' + self.update_dir + '/' + self.train_list[idx] + '.png')
            else:
                mask = Image.open(self.dataset_dir + '/masks_rand/' + self.train_list[idx] + '.png')
            mask = np.array(mask, dtype=np.float32)  / 255.0
        else:
            mask = self.mask_update_list[idx] 
            
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]

        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TrainSetLoader_coarse(Dataset):
    def __init__(self, dataset_dir, patch_size, update_dir='', mask_update_list=None):
        super(TrainSetLoader_coarse).__init__()
        self.dataset_dir = dataset_dir
        self.update_dir = update_dir
        if mask_update_list == None:
            if self.update_dir != '':
                if os.path.exists(dataset_dir + '/masks_coarse_update_' + self.update_dir + '/'):
                    shutil.rmtree(dataset_dir + '/masks_coarse_update_' + self.update_dir + '/')
                shutil.copytree(dataset_dir + '/masks_coarse/', dataset_dir + '/masks_coarse_update_' + self.update_dir + '/')
        self.mask_update_list = mask_update_list      
        self.patch_size = patch_size
        self.img_norm_cfg = dict(mean=95.010, std=41.511)
        with open(dataset_dir+'/50_50/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        if self.mask_update_list == None:
            if self.update_dir != '':
                mask = Image.open(self.dataset_dir + '/masks_coarse_update_' + self.update_dir + '/' + self.train_list[idx] + '.png')
            else:
                mask = Image.open(self.dataset_dir + '/masks_coarse/' + self.train_list[idx] + '.png')
            mask = np.array(mask, dtype=np.float32)  / 255.0
        else:
            mask = self.mask_update_list[idx]     
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]

        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)#C N H W
            target = target.transpose(1, 0)
        return input, target
