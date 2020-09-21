# -*- coding: utf-8 -*-
# filename: data_loader.py
# brief: load DRIVE dataset and form as tensors
# author: Jia Zhuang
# date: 2020-09-18

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob

def load_dataset(rel_path='.'):
    
    if os.path.exists("datasets/training/image.npy") and \
        os.path.exists("datasets/training/label.npy") and \
        os.path.exists("datasets/training/mask.npy"):
        
        new_input_tensor = np.load("datasets/training/image.npy")
        new_label_tensor = np.load("datasets/training/label.npy")
        new_mask_tensor = np.load("datasets/training/mask.npy")
        return new_input_tensor, new_label_tensor, new_mask_tensor

    train_image_files = sorted(glob(os.path.join(rel_path, 'DRIVE/training/images/*.tif')))
    train_label_files = sorted(glob(os.path.join(rel_path, 'DRIVE/training/1st_manual/*.gif')))
    train_mask_files = sorted(glob(os.path.join(rel_path, 'DRIVE/training/mask/*.gif')))
    
    for i, filename in enumerate(train_image_files):
        print('[*] adding {}th train image : {}'.format(i + 1, filename))
        img = Image.open(filename)
        imgmat = np.array(img).astype('float')
        imgmat = (imgmat / 255.0 - 0.5) * 2
        if i == 0:
            input_tensor = np.expand_dims(imgmat, axis=0)
        else:
            tmp = np.expand_dims(imgmat, axis=0)
            input_tensor = np.concatenate((input_tensor, tmp), axis=0)
    new_input_tensor = np.moveaxis(input_tensor, 3, 1)
            
    for i, filename in enumerate(train_label_files):
        print('[*] adding {}th train label : {}'.format(i + 1, filename))
        label = np.array(Image.open(filename))
        label = label / 255.0
        if i == 0:
            label_tensor = np.expand_dims(label, axis=0)
        else:
            tmp = np.expand_dims(label, axis=0)
            label_tensor = np.concatenate((label_tensor, tmp), axis=0)
    new_label_tensor = np.stack((label_tensor[:,:,:], 1 - label_tensor[:,:,:]), axis=1)
    
    for i, filename in enumerate(train_mask_files):
        print('[*] adding {}th train mask : {}'.format(i+1, filename))
        mask = np.array(Image.open(filename))
        mask = mask / 255.0
        if i == 0:
            mask_tensor = np.expand_dims(mask, axis=0)
        else:
            tmp = np.expand_dims(mask, axis=0)
            mask_tensor = np.concatenate((mask_tensor, tmp), axis=0)
    new_mask_tensor = np.stack((mask_tensor[:,:,:], mask_tensor[:,:,:]), axis=1)
    
    np.save("datasets/training/image.npy", new_input_tensor)
    np.save("datasets/training/label.npy", new_label_tensor)
    np.save("datasets/training/mask.npy", new_mask_tensor)
    
    return new_input_tensor, new_label_tensor, new_mask_tensor