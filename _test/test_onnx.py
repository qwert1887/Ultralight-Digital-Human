# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/10/30 16:24
import time

import cv2
import numpy as np
import onnx
import torch

from unet import Model

onnx_path = "dihuman.onnx"

def check_onnx(torch_out, torch_in, audio):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    import onnxruntime
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {ort_session.get_inputs()[0].name: torch_in.cpu().numpy(), ort_session.get_inputs()[1].name: audio.cpu().numpy()}
    for i in range(1):
        t1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        pred = ort_outs[0].squeeze()
        pred = pred.transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        # print(pred.shape, pred.dtype)
        t2 = time.time()
        print("onnx time cost::", t2 - t1)
        cv2.imwrite("test_zyz.jpg", pred)


if __name__ == '__main__':
    img_path = "dataset/zyz_0128/full_body_img/0.jpg"
    lms_path = "dataset/zyz_0128/landmarks/0.lms"
    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 150, 145), (0, 0, 0), -1)

    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None]
    net = Model(6).eval()
    net.load_state_dict(torch.load("checkpoint/hubert/195.pth"))
    # img = torch.zeros([1, 6, 160, 160])
    audio = torch.zeros([1, 32, 32, 32])
    # torch_out = net(img_concat_T, audio)
    torch_out = None
    check_onnx(torch_out, img_concat_T, audio)