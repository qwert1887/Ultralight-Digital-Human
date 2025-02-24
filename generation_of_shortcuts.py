# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2025/2/24 11:07
"""
训练后批量生成短视频平台的人物形象快照。包括任意一帧图片（如果是绿幕视频，
需要扣绿后做成透明背景的`.png`图片）、5s的训练视频素材。PS:注意model_id的命名，
0218_zyz_gs：
    `0128`代表02月18日拍摄的，
    `zyz`表示人名拼音首字母
    `gs`green screen缩写，绿幕视频，会影响到更换背景逻辑。

"""
import os
import subprocess
import shutil


def process_video(video_path, output_dir, model_id):

    # TODO 需要补充绿幕判断逻辑。或直接在文件夹中提前命名。目前全部默认绿幕视频
    model_id = f"{model_id}_gs"

    # 获取视频所在目录
    video_dir = os.path.dirname(video_path)

    # 生成输出文件名
    frame_output = os.path.join(video_dir, f"{model_id}.jpg")
    video_output = os.path.join(video_dir, f"{model_id}.mp4")
    target_frame_output = os.path.join(output_dir, f"{model_id}.jpg")  # .jpg OR .png
    target_video_output = os.path.join(output_dir, f"{model_id}.mp4")

    if not os.path.isfile(frame_output):
        # 提取第一帧图片
        subprocess.run([
            "ffmpeg","-loglevel", "warning", "-hide_banner", "-i", video_path, "-vf", "select=eq(n\,0)", "-q:v", "2", frame_output
        ], check=True)
    if not os.path.isfile(video_output):
        # 提取前5秒视频（不包含音频）
        subprocess.run([
            "ffmpeg", "-loglevel", "warning", "-hide_banner","-i", video_path, "-t", "5", "-an", video_output
        ], check=True)

    # 拷贝文件到指定目录
    if not os.path.isfile(target_frame_output):
        shutil.copy(frame_output, output_dir)
    if not os.path.isfile(target_video_output):
        shutil.copy(video_output, output_dir)


def process_folder(folder_path, output_dir):
    # 遍历文件夹
    for model_id in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, model_id)):
            continue
        video_path = os.path.join(folder_path, model_id, f"1080p.mp4")
        if not os.path.isfile(video_path):
            print(f"The video {video_path} is not exist!!!")
            continue
        process_video(video_path, output_dir, model_id)


if __name__ == "__main__":
    # 设置要遍历的文件夹路径和输出目录
    folder_path = "dataset"
    # video_path = "dataset/0123_zhou_stand/1080p.mp4"
    output_dir = "materials"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理文件夹
    process_folder(folder_path, output_dir)
    # process_video(video_path, output_dir, "0123_zhou_stand_gs")
