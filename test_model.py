# -*- coding: utf-8 -*-
# filename: test_model.py
# brief: test U-net model on DRIVE dataset
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from unet_model import Unet
from utils import paste_and_save, eval_print_metrics
from data_loader import load_dataset


def model_test(net, batch_size=2):
    
    x_tensor, y_tensor, m_tensor = load_dataset(mode='test', resize=True, resize_shape=(256, 256))
    num_samples = x_tensor.shape[0]
    print("[+] ====== Start test... ======")
    num_iters = int(np.ceil(num_samples / batch_size))
    for ite in range(num_iters):
        print("[*] predicting on the {}th batch".format(ite + 1))
        if not ite == num_iters - 1:
            start_id, end_id = ite * batch_size, (ite + 1) * batch_size
            bat_img = torch.Tensor(x_tensor[start_id : end_id, :, :, :])
            bat_label = torch.Tensor(y_tensor[start_id : end_id, 0: 1, :, :])
            #bat_mask_2ch = torch.Tensor(m_tensor[start_id : end_id, :, :, :])
            bat_mask = torch.Tensor(m_tensor[start_id : end_id, 0: 1, :, :])
        else:
            start_id = ite * batch_size
            bat_img = torch.Tensor(x_tensor[start_id : , :, :, :])
            bat_label = torch.Tensor(y_tensor[start_id : , 0: 1, :, :])
            #bat_mask_2ch = torch.Tensor(m_tensor[start_id : end_id, :, :, :])
            bat_mask = torch.Tensor(m_tensor[start_id : , 0: 1, :, :])
        bat_pred = net(bat_img)
        bat_pred_class = (bat_pred > 0.5).float() * bat_mask
        eval_print_metrics(bat_label, bat_pred, bat_mask)
        # plt.imshow(bat_pred[0,0,:,:].detach().numpy(), cmap='jet')#, vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        #bat_pred_class = bat_pred.detach() * bat_mask
        paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, ite + 1)

    return


if __name__ == "__main__":
    if not os.path.exists("./pred_imgs"):
        os.mkdir("./pred_imgs")
    if not os.path.exists("./datasets/test"):
        os.mkdir("./datasets/test")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected_model = glob("./checkpoint/Unet_epoch*.model")[-1]
    print("[*] Selected model for testing: {} ".format(selected_model))
    unet_ins = Unet(img_ch=3, isDeconv=True, isBN=True)
    unet_ins.load_state_dict(torch.load(selected_model))
    unet_ins.to(device)
    model_test(unet_ins, batch_size=2)
    