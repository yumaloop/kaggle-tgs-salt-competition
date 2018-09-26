import numpy as np
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


def upsample(img, img_size_ori=101, img_size_target=101):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
    
def downsample(img, img_size_ori=101, img_size_target=101):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i: 
            return i