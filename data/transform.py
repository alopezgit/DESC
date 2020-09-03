import collections
import math
import numbers
import random

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class RandomHorizontalFlip(object):
    """
    Random horizontal flip.

    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, img):
        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        
        return img


class RandomVerticalFlip(object):
    """
    Random vertical flip.

    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob
        
    def __call__(self, img):

        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomPairedCrop(object):

    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        """
        Get parameters for ``crop`` for a random crop.
        Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
        Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        img1 = img[0]
        img2 = img[1]
        depth = img[2] 
        
        i, j, th, tw = self.get_params(img1, self.size)

        img1 = F.crop(img1, i, j, th, tw)

        if depth is not None:
            depth = F.crop(depth, i, j, th, tw)
        if img2 is not None:
            img2 = F.crop(img2, i, j, th, tw)
        return img1, img2, depth 

        
class RandomImgAugment(object):
    """Randomly shift gamma"""

    def __init__(self, size=None):
        self.size = size

    def __call__(self, inputs):

        img1 = inputs[0]
        img2 = inputs[1]
        depth = inputs[2]
        phase = inputs[3]
        fb = inputs[4]

        h = img1.height
        w = img1.width
        w0 = w

        if self.size == [-1]:
            divisor = 32.0
            h = int(math.ceil(h/divisor) * divisor)
            w = int(math.ceil(w/divisor) * divisor)
            self.size = (h, w)
       
        scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
        
        img1 = scale_transform(img1)
        if img2 is not None:
            img2 = scale_transform(img2)
        
        if fb is not None:
            # Scaling of the focal for e.g. stereo photo loss
            scale = float(self.size[1]) / float(w0)
            fb = fb * scale

        if phase == 'test':
            return img1, img2, depth, fb
    
        if depth is not None:
           scale_transform_d = transforms.Compose([transforms.Resize(self.size, Image.NEAREST)])
           depth = scale_transform_d(depth)

        if not self.size == 0:

            if depth is not None:
                # This maps the VKITTI depth to [0, 1] 
                # with 1 being 655.35 meters
                arr_depth = np.array(depth, dtype=np.float32)
                arr_depth /= 65535.0  # cm->m
                arr_depth[arr_depth<0.0] = 0.0
                depth = Image.fromarray(arr_depth, 'F')

        if depth is not None:
            # Maps depth to [-1, 1] to match tanh output of 
            # the depth model
            depth = np.array(depth, dtype=np.float32)   
            depth = depth * 2.0
            depth -= 1.0    

        if random.random() < 0.5:
            # Some brightness, contrast and saturation augmentation
            brightness = random.uniform(0.8, 1.0)
            contrast = random.uniform(0.8, 1.0)
            saturation = random.uniform(0.8, 1.0)

            img1 = F.adjust_brightness(img1, brightness)
            img1 = F.adjust_contrast(img1, contrast)
            img1 = F.adjust_saturation(img1, saturation)
        
            if img2 is not None:
                img2 = F.adjust_brightness(img2, brightness)
                img2 = F.adjust_contrast(img2, contrast)
                img2 = F.adjust_saturation(img2, saturation)    
        return img1, img2, depth, fb

class DepthToTensor(object):
    def __call__(self, input):

        arr_input = np.array(input)
        tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1]))).float()
        return tensors

