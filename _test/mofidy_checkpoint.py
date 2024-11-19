# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/15 19:32
import os

import torch
device = "cuda" if torch.cuda.is_available() else 'cpu'
ckpt_path = "checkpoints/huizhang_1106/130.pth"
if not os.path.isfile(ckpt_path):
    raise ValueError(f"Checkpoint {ckpt_path} doesn't exist!")
checkpoint_dict = torch.load(ckpt_path, map_location=device)
print(checkpoint_dict.keys())

checkpoint_dict["epoch"] = 130
# start_epoch = checkpoint_dict['epoch']
# start_lr = checkpoint_dict['lr']
# avg_loss = checkpoint_dict['avg_loss']
# optimizer = checkpoint_dict["optimizer"]
# model = checkpoint_dict["model"]

torch.save(checkpoint_dict, os.path.join(os.path.dirname(ckpt_path), "130_new3.pth"))
