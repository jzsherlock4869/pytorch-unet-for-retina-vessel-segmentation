# -*- coding: utf-8 -*-
# filename: utils.py
# brief: some utility functions
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

def eval_print_metrics(bat_label, bat_pred, bat_mask):
    assert len(bat_label.size()) == 4 \
        and len(bat_pred.size()) == 4 and len(bat_mask.size()) == 4
    assert bat_label.size()[1] == 1 \
        and bat_pred.size()[1] == 1 and bat_mask.size()[1] == 1
    
    masked_pred = bat_pred * bat_mask
    masked_label = bat_label * bat_mask
    masked_pred_class = (masked_pred > 0.5).float()
    
    precision = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_pred_class)) + 1)
    recall = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_label.float())) + 1)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)
    
    pred_ls = np.array(bat_pred[bat_mask > 0].detach().cpu())
    label_ls = np.array(bat_label[bat_mask > 0].detach().cpu(), dtype=int)
    bat_auc = roc_auc_score(label_ls, pred_ls)
    bat_roc = roc_curve(label_ls, pred_ls)

    print("[*] ...... Evaluation ...... ")
    print(" >>> precision: {:.4f} recall: {:.4f} f1_score: {:.4f} auc: {:.4f}".format(precision, recall, f1_score, bat_auc))

    return precision, recall, f1_score, bat_auc, bat_roc

def paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, cur_bat_num, save_img='pred_imgs'):
    w, h = bat_img.size()[2:4]
    for bat_id in range(bat_img.size()[0]):
        # img = Image.fromarray(np.moveaxis(np.array((bat_img + 1) / 2 * 255.0, dtype=np.uint8)[bat_id, :, :, :], 0, 2))
        img = Image.fromarray(np.moveaxis(np.array(bat_img.cpu() * 255.0, dtype=np.uint8)[bat_id, :, :, :], 0, 2))
        label = Image.fromarray(np.array(bat_label.cpu() * 255.0, dtype=np.uint8)[bat_id, 0, :, :])
        pred_class = Image.fromarray(np.array(bat_pred_class.cpu() * 255.0, dtype=np.uint8)[bat_id, 0, :, :])
        
        res_id = (cur_bat_num - 1) * batch_size + bat_id
        target = Image.new('RGB', (3 * w, h))
        target.paste(img, box = (0, 0))
        target.paste(label, box = (w, 0))
        target.paste(pred_class, box = (2 * w, 0))
        
        target.save(os.path.join(save_img, "result_{}.png".format(res_id)))
    return

