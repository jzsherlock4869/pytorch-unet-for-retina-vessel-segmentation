# -*- coding: utf-8 -*-
# filename: utils.py
# brief: some utility functions
# author: Jia Zhuang
# date: 2020-09-21

from PIL import Image
import os

def paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, cur_bat_num, save_img='pred_imgs'):
    w, h = bat_img.size()[2:4]
    for bat_id in range(bat_img.size()[0]):
        img = Image.fromarray(bat_img[bat_id])
        label = Image.fromarray(bat_label[bat_id])
        pred_class = Image.fromarray(bat_pred_class[bat_id])
        
        res_id = (cur_bat_num - 1) * batch_size + bat_id
        target = Image.new('RGB', (3 * w, h))
        target.paste(img, box = (0, 0))
        target.paste(label, box = (w, 0))
        target.paste(pred_class, box = (2 * w, 0))
        
        target.save(os.path.join(save_img, "result_{}.png".format(res_id)))
    return

