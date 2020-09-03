import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data
from utils.dataset_util import KITTI
import random
import cv2
import copy
import json
import torchvision
import imageio
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):

        dd = {}
        {dd.update(d[i]) for d in self.datasets if d is not None}
        
        return dd

    def __len__(self):
        return max(len(d) for d in self.datasets if d is not None)

class VKittiDataset(data.Dataset):
    def __init__(self, root='./datasets', data_file='train.txt',
                 phase='train', img_transform=None, depth_transform=None,
                 joint_transform=None):
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        self.joint_transform = joint_transform
        self.to_tensor = torchvision.transforms.ToTensor()

        with open(osp.join('./datasets/vkitti/', self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:

                if len(data) == 0:
                    continue
                data_info = data.split(' ')                
                
                self.files.append({
                    "rgb": data_info[0],
                    "depth": data_info[1]
                    })
                
                
                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
        assert osp.exists(osp.join(self.root, datafiles['rgb'])), "Image {:s} does not exist".format(datafiles['rgb'])
        rgb = Image.open(osp.join(self.root, datafiles['rgb'])).convert('RGB')
        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth {:s} does not exist".format(datafiles['depth'])                
        depth = Image.open(osp.join(self.root, datafiles['depth']))      

        return rgb, depth
    
    def __getitem__(self, index):
        if index > len(self) - 1:
            index = index % len(self)
        datafiles = self.files[index]
        img, depth, = self.read_data(datafiles)
        original_img = copy.deepcopy(img)

        # VKitti focal
        fb_or = 725
        if self.joint_transform is not None:
            if self.phase == 'train':    
                img, _, depth, fb = self.joint_transform((img, None, depth, self.phase, fb_or))
            else:
                img, _, depth, fb = self.joint_transform((img, None, depth, 'test', fb_or))
        
        # This resize of the original resolution image should not be needed anymore with detectron2. We used 
        # to have another model that only accepted image sizes divible by 16, hence why we resized
        # to this 1232x368. We just leave it to match the pipeline used for the paper.
        resize = 1232/self.to_tensor(original_img).shape[-1]
        original_img = torch.nn.functional.interpolate(self.to_tensor(original_img).unsqueeze(0), size=(368, 1232)).squeeze(0)
        fb_or = resize * fb_or
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        if self.phase =='test':
            data = {}
            data['id'] = index    
            data['img'] = img
            data['depth'] = depth
            data['fb'] = fb
            data['focal'] = fb_or
            data['original_img'] = original_img
            data['id'] = index
            data['name'] = 'vkitti'
            return data

        data = {}
        if img is not None:
            data['img'] = img
            data['original_img'] = original_img
        if depth is not None:
            data['depth'] = depth
        if fb is not None:
            data['fb'] = fb
            data['focal'] = fb_or

        data['id'] = index
        data['name'] = 'vkitti'    

        return {'src': data}


class KittiDataset(data.Dataset):
    def __init__(self, root='./datasets', data_file='train.txt', phase='train',
                 img_transform=None, joint_transform=None, depth_transform=None):
      
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.depth_transform = depth_transform
        self.to_tensor = torchvision.transforms.ToTensor()
        with open(osp.join('./datasets/kitti/', self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                self.files.append({
                    "l_rgb": data_info[0],
                    "r_rgb": data_info[1],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3]
                    })
                                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
        assert osp.exists(osp.join(self.root, datafiles['l_rgb'])), "Image {:s} does not exist".format(datafiles['l_rgb'])
        l_rgb = Image.open(osp.join(self.root, datafiles['l_rgb'])).convert('RGB')
        w = l_rgb.size[0]
        h = l_rgb.size[1]
        assert osp.exists(osp.join(self.root, datafiles['r_rgb'])), "Image {:s} does not exist".format(datafiles['r_rgb'])
        r_rgb = Image.open(osp.join(self.root, datafiles['r_rgb'])).convert('RGB')

        kitti = KITTI()
        assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        fb, focal = kitti.get_fb(osp.join(self.root, datafiles['cam_intrin']))
        assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth {:s} does not exist".format(datafiles['depth'])
        depth = kitti.get_depth(osp.join(self.root, datafiles['cam_intrin']),
                                osp.join(self.root, datafiles['depth']), [h, w])
        

        return l_rgb, r_rgb, fb, focal, depth
    
    def __getitem__(self, index):

        if index > len(self)-1:
            index = index % len(self)
        datafiles = self.files[index]
        l_img, r_img, fb, focal, depth = self.read_data(datafiles)

        original_l_img = copy.deepcopy(l_img)
        original_r_img = copy.deepcopy(r_img)
        fb_or = fb

        if self.joint_transform is not None:
            if self.phase == 'train':
                l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'train', fb))
            else:
                l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'test', fb))

        # This resize of the original resolution image should not be needed anymore with detectron2. We used 
        # to have another model that only accepted image sizes divible by 16, hence why we resized
        # to this 1232x368. We just leave it to match the pipeline used for the paper.
        resize = 1232/self.to_tensor(original_l_img).shape[-1]
        original_l_img = torch.nn.functional.interpolate(self.to_tensor(original_l_img).unsqueeze(0), size=(368, 1232)).squeeze(0)
        original_r_img = torch.nn.functional.interpolate(self.to_tensor(original_r_img).unsqueeze(0), size=(368, 1232)).squeeze(0) 
        focal = resize * focal       
        if self.img_transform is not None:
            l_img = self.img_transform(l_img)
            if r_img is not None:
                r_img = self.img_transform(r_img)
          
        
        if self.phase =='test':
            data = {}
            data['left_img'] = l_img
            data['original_left_img'] = original_l_img
            data['right_img'] = r_img
            data['original_right_img'] = original_r_img
            data['depth'] = depth
            data['fb'] = fb
            data['focal'] = focal
            data['id'] = index
            data['name'] = 'kitti'   
            return data

        data = {}
        if l_img is not None:
            data['left_img'] = l_img
            data['original_left_img'] = original_l_img
        if r_img is not None:
            data['right_img'] = r_img
            data['original_right_img'] = original_r_img

        if fb is not None:
            data['fb'] = fb
        if focal is not None:
            data['focal'] = focal
        if depth is not None:
            depth = cv2.resize(depth, dsize=(1232, 368), interpolation=cv2.INTER_NEAREST)
            data['depth'] = depth
        data['id'] = index
        data['name'] = 'kitti'    
        return {'tgt': data}

def get_dataset(root, data_file='train.list', dataset='kitti', phase='train',
                img_transform=None, depth_transform=None,
                joint_transform=None, test_dataset='kitti'):

    DEFINED_DATASET = {'KITTI', 'VKITTI'}
    assert dataset.upper() in DEFINED_DATASET
    name2obj = {'KITTI': KittiDataset,
                'VKITTI': VKittiDataset}

    return name2obj[dataset.upper()](root=root, data_file=data_file, phase=phase,
                                     img_transform=img_transform, depth_transform=depth_transform,
                                     joint_transform=joint_transform)
