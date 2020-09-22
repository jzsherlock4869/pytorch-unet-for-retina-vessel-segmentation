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
from utils import eval_print_metrics
import warnings
warnings.filterwarnings('ignore')

def model_train(net, epochs=500, batch_size=2, lr=1e-2, save_every=5, eval_every=10, is_eval=True):
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #loss_func = nn.CrossEntropyLoss()
    x_tensor, y_tensor, m_tensor = load_dataset(mode="training", resize=True, resize_shape=(256, 256))
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
                bat_label = torch.Tensor(y_tensor[start_id : end_id, 0: 1, :, :])
                # bat_mask_2ch = torch.Tensor(m_tensor[start_id : end_id, :, :, :])
                bat_mask = torch.Tensor(m_tensor[start_id : end_id, 0: 1, :, :])
            else:
                start_id = ite * batch_size
                bat_img = torch.Tensor(x_tensor[start_id : , :, :, :])
                bat_label = torch.Tensor(y_tensor[start_id : , 0: 1, :, :])
                # bat_mask_2ch = torch.Tensor(m_tensor[start_id : , :, :, :])
                bat_mask = torch.Tensor(m_tensor[start_id : , 0: 1, :, :])
            criterion = nn.BCELoss()
            optimizer.zero_grad()
            bat_pred = net(bat_img)
            # print(bat_pred.size(), bat_mask.size(), bat_label.size())
            # masked_pred = bat_pred * bat_mask
            # masked_label = bat_label * bat_mask
            loss = criterion(bat_pred, bat_label.float())
            # loss = loss_func(masked_pred, masked_label.float())
            print("[*] Epoch: {}, Iter: {} current loss: {:.8f}"\
                  .format(epoch + 1, ite + 1, loss.item()))
            if is_eval:
                if epoch % eval_every == 0:
                    print("[*] ...... Eval for Epoch: {}, Iter: {} ....... ".format(epoch + 1, ite + 1))
                    eval_print_metrics(bat_label, bat_pred, bat_mask)

            if not ite == num_iters - 1:
                epoch_tot_loss += loss.item()
            else:
                epoch_tot_loss += loss.item()
                epoch_avg_loss = epoch_tot_loss / (ite + 1)
                print("[+] ====== Epoch {} finished, avg_loss : {:.8f} ======"\
                      .format(epoch + 1, epoch_avg_loss))
            loss.backward()
            optimizer.step()
        if epoch % save_every == 0:
            torch.save(net.state_dict(), "./checkpoint/Unet_epoch{}_loss{:.4f}_retina.model".format(str(epoch + 1).zfill(5), epoch_avg_loss))
    return net

if __name__ == "__main__":

    if not os.path.exists("./checkpoint"):
        os.mkdir("./checkpoint")
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    if not os.path.exists("./datasets/training"):
        os.mkdir("./datasets/training")

    # set parameters for training
    LR = 0.1
    EPOCHS = 500
    BATCH_SIZE = 8
    SAVE_EVERY = 20
    EVAL_EVERY = 30

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_ins = Unet(img_ch=3, isDeconv=True, isBN=True)
    unet_ins.to(device)
    trained_unet = model_train(unet_ins, 
                            batch_size=BATCH_SIZE, 
                            lr=LR, 
                            epochs=EPOCHS, 
                            save_every=SAVE_EVERY, 
                            eval_every=EVAL_EVERY)
