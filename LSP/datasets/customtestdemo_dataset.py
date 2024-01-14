from operator import gt
import os
from datasets.base_dataset import BaseDataset
import os.path
from pathlib import Path
import torch
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from PIL import Image
import bisect
import numpy as np
import io
import cv2
import h5py
import albumentations as A
import albumentations.pytorch as AP
import glob
import PIL
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import pdb

    
class CustomTestDemoDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.state = 'Test'
        self.dataset_name = opt.dataset_names[0]
        self.opt = opt
        # only load in train mode
        self.dataset_root = os.path.join(self.root, self.dataset_name)

        # gt_img_path = glob.glob(self.root+'/origin/*.png')
        # gt_img_path = [path for path in gt_img_path if '[No_Face]' not in path]
        # gt_img_path = sorted(gt_img_path, key = lambda x: int(x.split('/')[-1].split('.')[0]))
        # gt_feature_path = glob.glob(self.root+'/FaceMesh/*.png')
        # gt_feature_path = sorted(gt_feature_path, key = lambda x: int(x.split('/')[-1].split('.')[0]))
        # gt_feature_path = [path.replace('Origin', 'FaceMesh') for path in gt_img_path]
        print(f"!!!{self.root}")
        gt_feature_path = glob.glob(self.root+'/facemesh_shoulder/*.png')
        gt_feature_path = sorted(gt_feature_path, key = lambda x: int(x.split('/')[-1].split('.')[0]))

        self.gt_feature_path = gt_feature_path

        self.transform = A.Compose([
                A.Resize(self.opt.loadSize_h, self.opt.loadSize_w),
                AP.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})])

        # tmp = []
        # for j in range(4):
        #     output = imread(self.root+'/candidates/normalized_full_{}.png'.format(j))
        #     # output = self.transform(output)
        #     output = AP.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
        #     tmp.append(output)
        # self.cand = torch.cat(tmp)

        # print(len(self.gt_img_path))
        # print(len(self.gt_feature_path))
        

    def __len__(self):
        return len(self.gt_feature_path)

    def name(self):
        return 'CustomDatasetTestDemo'
            
    
    def __getitem__(self, ind):

        # data_index = ind * self.opt.frame_jump + np.random.randint(self.opt.frame_jump)
        target_ind = ind
        
        # if target_ind < 2:
        #     target_ind +=2
        # elif target_ind + 2 >= len(self.gt_feature_path)-1:
        #     target_ind -= 2
        
        feature_file_path = self.gt_feature_path[target_ind]
        feature_map = imread(feature_file_path)
        feature_map = resize(feature_map, (self.opt.loadSize_h, self.opt.loadSize_w))
        feature_map = rgb2gray(feature_map)
        feature_map = np.array(feature_map).astype(np.float32)
        feature_map = feature_map.reshape(1, self.opt.loadSize_h, self.opt.loadSize_w)
        feature_map = torch.from_numpy(feature_map)
        
        tmp = []
        if target_ind < 2 or target_ind + 2 > len(self.gt_feature_path)-1:
            for i in range(4):
                tmp.append(feature_map)
            cand_image = torch.cat(tmp)
        else:
            for i in range(-2,3):
                if i==0:
                    continue
                output = imread(self.gt_feature_path[target_ind-i])
                output = resize(output, (self.opt.loadSize_h, self.opt.loadSize_w))
                output = rgb2gray(output)
                output = np.array(output).astype(np.float32)
                output = output.reshape(1, self.opt.loadSize_h, self.opt.loadSize_w)
                output = torch.from_numpy(output)
                tmp.append(output)
            cand_image = torch.cat(tmp)
        
        return_list = {'feature_map': feature_map, 'cand_image': cand_image, 'tgt_image': feature_map, 'weight_mask': feature_map, 'cand_feature': feature_map}
           
        return return_list
  
    

