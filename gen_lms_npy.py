# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/6 16:09
import os.path
import pickle
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def gen_lms_pickle(dataset_dir):
    """ 根据数据目录生成lms缓存数据 """

    video_path = os.path.join(dataset_dir, '1080p.mp4')  # 数据下面有1080p的训练视频，最好是无声视频
    lms_dir = os.path.join(dataset_dir, 'landmarks')
    cache_path = f"{os.path.join(dataset_dir, 'lms_cache_data.pkl')}"
    if not os.path.isdir(lms_dir) or os.path.isfile(cache_path):
        print(f"Landmarks dir {lms_dir} not found OR {cache_path} is exist!")
        return
    # img = cv2.imread(img_path)
    face_dict = {}
    crop_face_ori_list = []
    crop_face_list = []
    coord_list = []  # [(xmin, ymin, xmax, ymax, w, h), ...]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p_bar = tqdm(total=total_frames, ncols=100)
    if not cap.isOpened():
        raise IOError(f'Cannot open video file {video_path}')
    n = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        lms_list = []
        with open(f"{os.path.join(lms_dir, f'{n}.lms')}", "r") as f:
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
        coord_list.append([xmin, ymin, xmax, ymax, w, h])
        p_bar.update(1)
        n += 1

    # face_dict['concat_face_mask'] = crop_face_list
    # face_dict['crop_face_ori'] = crop_face_ori_list
    face_dict['coord_list'] = coord_list
    with open(cache_path, "wb") as f:
        pickle.dump(face_dict, f)


def gen_img_concat_mask(video_path):
    # img = cv2.imread(img_path)
    face_dict = {}
    crop_face_ori_list = []
    crop_face_list = []
    coord_list = []  # [(xmin, ymin, xmax, ymax, w, h), ...]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p_bar = tqdm(total=total_frames, ncols=100)
    if not cap.isOpened():
        raise IOError(f'Cannot open video file {video_path}')
    n = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        lms_list = []
        with open(f"landmarks/{n}.lms", "r") as f:
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
        crop_face_list.append(img_concat_T)
        crop_face_ori_list.append(crop_img_ori)
        coord_list.append([xmin, ymin, xmax, ymax, w, h])
        p_bar.update(1)
        n += 1

    # face_dict['concat_face_mask'] = crop_face_list
    # face_dict['crop_face_ori'] = crop_face_ori_list
    face_dict['coord_list'] = coord_list
    with open("cache_data.pkl", "wb") as f:
        pickle.dump(face_dict, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)
        # print(loaded_dict)
    print(loaded_dict["coord_list"])


if __name__ == '__main__':
    # gen_img_concat_mask("1080p.mp4")
    # load_pickle("cache_data.pkl")
    # dataset_dir = './dataset/0123_girl_a_sit'
    # gen_lms_pickle(dataset_dir)
    dataset_dir = './dataset'
    for file_dir in os.listdir(dataset_dir):
        print(file_dir)
        spec_dataset = os.path.join(dataset_dir, file_dir)
        if not os.path.isdir(spec_dataset):
            print(f"{spec_dataset} 非文件目录，跳过!")
            continue
        gen_lms_pickle(spec_dataset)
