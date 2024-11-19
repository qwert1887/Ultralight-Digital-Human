import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--preload', action='store_true', help="it determines if we preload the data into memory")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=16)  # default 32
    parser.add_argument('--lr', type=float, default=0.001)  # default 0.001
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--resume_ckpt', type=str, default="")

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet
# Loss functions
class PerceptualLoss():
    
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(net, epoch, batch_size, lr):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    content_loss = PerceptualLoss(torch.nn.MSELoss())  # default
    # content_loss = PerceptualLoss(torch.nn.L1Loss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")
            
        syncnet = SyncNet_color(args.asr).eval().cuda()
        sync_dict_state = torch.load(args.syncnet_checkpoint)
        if "model" in sync_dict_state:
            syncnet.load_state_dict(sync_dict_state["model"])
        else:
            syncnet.load_state_dict(sync_dict_state)
    save_dir= args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataloader_list = []
    dataset_list = []
    dataset_dir_list = [args.dataset_dir]
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir, args.asr, preload=args.preload)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)
    start_epoch = 0
    # start_lr = lr
    avg_loss = 0.0
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if args.resume_ckpt != "":
        if not os.path.isfile(args.resume_ckpt):
            raise ValueError(f"Checkpoint {args.resume_ckpt} doesn't exist!")
        checkpoint_dict = torch.load(args.resume_ckpt, map_location=device)
        # print(checkpoint_dict.keys())
        if 'model' in checkpoint_dict:
            missing_keys, unexpected_keys = net.load_state_dict(checkpoint_dict['model'], strict=False)
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print(f"missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
            print(f"[INFO] loaded model.")
        if 'optimizer' in checkpoint_dict:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            print("[INFO] loaded optimizer.")
        start_epoch = checkpoint_dict['epoch']
        lr = checkpoint_dict['lr']
        avg_loss = checkpoint_dict['avg_loss']
    criterion = nn.L1Loss()  # default
    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    lr_stop_reduce = True
    for e in range(1, epoch+1):
        e = e + start_epoch
        net.train()
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]
        
        with tqdm(total=len(dataset), ncols=100, desc=f'Epoch {e}/{epoch+start_epoch}', unit=' img') as p:
            loss_list = []
            if e != 0 and e % 20 == 0 and not lr_stop_reduce:
                lr = max(lr * 0.5, 1e-4)
                lr_stop_reduce = lr == 1e-4
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                preds = net(imgs, audio_feat)
                if use_syncnet:
                    y = torch.ones([preds.shape[0],1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                if use_syncnet:
                    loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss  # default 10*sync_loss
                else:
                    loss = loss_pixel + loss_PerceptualLoss*0.01
                loss_value = loss.item()
                loss_list.append(loss_value)
                p.set_postfix(**{'loss (batch)': loss_value, "lr": lr})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])
            average_loss = sum(loss_list) / len(loss_list)
            p.set_postfix(**{'loss (batch)': average_loss, "lr": lr})
        if e % 10 == 0:
            state = {
                'epoch': e,
                'avg_loss': average_loss,
                'lr': lr,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(save_dir, str(e)+'.pth'))
        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+"_real.jpg", img_real)
        
            

if __name__ == '__main__':
    
    
    net = Model(6, mode=args.asr).cuda()
    # net.load_state_dict(torch.load("/usr/anqi/dihuman/checkpoint_female4/3070.pth"))
    train(net, args.epochs, args.batchsize, args.lr)