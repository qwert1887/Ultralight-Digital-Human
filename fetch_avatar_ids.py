# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2025/2/24 14:13
"""
分别从checkpoints、dataset中获取model.onnx、lms_cache_data.pkl和1080p_s.mp4文件，用于短视频生成平台中的单人形象生成。
其中avatar_id命名规则如下：eg.0122_zs_gs，`0122`表示01月22号拍摄，`zs`表示拍摄人物名字字母缩写，`gs`代表是绿幕视频[该标志决定业务逻辑中是否扣绿幕的逻辑]。
"""
import os
import shutil
import subprocess

dataset_dir = 'dataset'
checkpoints_dir = 'checkpoints'
output_dir = 'avatar_ids'

os.makedirs(output_dir, exist_ok=True)


for avatar_id in os.listdir(dataset_dir):
    print(avatar_id)
    output_avatar_path = os.path.join(output_dir, avatar_id)
    video_path = os.path.join(dataset_dir, avatar_id, "1080p.mp4")
    if not os.path.isfile(video_path):
        continue
    os.makedirs(output_avatar_path, exist_ok=True)
    target_video_path = f"{output_avatar_path}/1080p_s.mp4"
    if not os.path.isfile(target_video_path):
        # 生成1080p_s.mp4
        subprocess.run([
            "ffmpeg", "-loglevel", "warning", "-hide_banner","-i", video_path, "-an", "-y", f"{target_video_path}"
        ], check=True)
    # 从checkpoints中拷贝model.onnx
    onnx_path = f"{checkpoints_dir}/{avatar_id}/model.onnx"
    if not os.path.isfile(f"{output_avatar_path}/model.onnx"):
        assert os.path.isfile(onnx_path), f"{onnx_path} does not exist!!!"
        # print(f"{onnx_path} does not exist!!!")
        shutil.copy(onnx_path, output_avatar_path)
    # 从dataset目录中拷贝lms_cache_data.pkl文件
    lms_cache_path = f"{dataset_dir}/{avatar_id}/lms_cache_data.pkl"
    if not os.path.isfile(f"{output_avatar_path}/lms_cache_data.pkl"):
        assert os.path.isfile(lms_cache_path), f"{lms_cache_path} does not exist!!!"
        # print(f"{lms_cache_path} does not exist!!!")
        shutil.copy(lms_cache_path, output_avatar_path)