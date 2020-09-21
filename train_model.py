# -*- coding: utf-8 -*-
# filename: train_model.py
# brief: train U-net model on DRIVE dataset
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
from data_loader import load_dataset
from unet_model import Unet

def model_train(net, epochs=500, batch_size=2, lr=0.01):
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    x_tensor, y_tensor, m_tensor = load_dataset()
    #x_tensor, y_tensor, m_tensor = rim_padding(x_tensor), rim_padding(y_tensor), rim_padding(m_tensor)
    num_samples = x_tensor.shape[0]
    for epoch in range(epochs):
        epoch_tot_loss = 0
        print("[+] ====== Start training... epoch {} ======".format(epoch + 1))
        num_iters = int(np.ceil(num_samples / batch_size))
        shuffle_ids = np.random.permutation(num_samples)
        x_tensor = x_tensor[shuffle_ids, :, :, :]
        y_tensor = y_tensor[shuffle_ids, :, :, :]
        m_tensor = m_tensor[shuffle_ids, :, :, :]
        for ite in range(num_iters):
            if not ite == num_iters - 1:
                start_id, end_id = ite * batch_size, (ite + 1) * batch_size
                bat_img = torch.Tensor(x_tensor[start_id : end_id, :, :, :])
                bat_label = torch.Tensor(y_tensor[start_id : end_id, :, :])
                bat_mask = torch.Tensor(m_tensor[start_id : end_id, :, :])
            else:
                start_id = ite * batch_size
                bat_img = torch.Tensor(x_tensor[start_id : , :, :, :])
                bat_label = torch.Tensor(y_tensor[start_id : , :, :])
                bat_mask = torch.Tensor(m_tensor[start_id : , :, :])
            optimizer.zero_grad()
            bat_pred = net(bat_img)
            loss = loss_func(bat_pred * bat_mask, bat_label * bat_mask)
            print("[*] Epoch: {}, Iter: {} current loss: {:.8f}"\
                  .format(epoch + 1, ite + 1, loss.item()))
            if not ite == num_iters - 1:
                epoch_tot_loss += loss.item()
            else:
                epoch_tot_loss += loss.item()
                epoch_avg_loss = epoch_tot_loss / (ite + 1)
                print("[+] ====== Epoch {} finished, avg_loss : {:.8f} ======"\
                      .format(epoch + 1, epoch_avg_loss))
            loss.backward()
            optimizer.step()
        torch.save(net, "./checkpoint/Unet_epoch{}_loss{:.4f}_retina.model".format(epoch + 1, epoch_avg_loss))
    return net

if __name__ == "__main__":
    
    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    if not os.path.exists("./datasets/training"):
        os.mkdir("./datasets/training")
    if not os.path.exists("./datasets/testing"):
        os.mkdir("./datasets/testing")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_ins = Unet(img_ch=3, K_class=2)
    unet_ins.to(device)
    trained_unet = model_train(unet_ins, batch_size=5, epochs=50)