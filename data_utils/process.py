import os
import sys

from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import argparse
import numpy as np

def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -loglevel warning -hide_banner -c:a pcm_s16le -f wav -ar {sample_rate} -ac 1 {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path, mode):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS:{fps}, frames:{total_frames}")
    if mode == "hubert" and fps != 25:
        raise ValueError("Using hubert,your video fps should be 25!!!")
    if mode == "wenet" and fps != 20:
        raise ValueError("Using wenet,your video fps should be 20!!!")
        
    print("extracting images...")
    pbar = tqdm(total=total_frames, ncols=100)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
        pbar.update(1)
        
def get_audio_feature(wav_path, mode):
    
    print("extracting audio feature...")
    
    if mode == "wenet":
        os.system("python wenet_infer.py "+wav_path)
    if mode == "hubert":
        os.system("python hubert.py --wav "+wav_path)
    
def get_landmark(path, landmarks_dir):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
    
    from get_landmark import Landmark
    landmark = Landmark()
    images_names = os.listdir(full_img_dir)
    temp_img = os.path.join(full_img_dir, images_names[0])
    h, w = cv2.imread(temp_img).shape[:2]
    for img_name in tqdm(images_names, ncols=100):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = max(0, p[0]+x1), max(0, p[1]+y1)  # border
                x, y = min(w, x), min(h, y)
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--asr', type=str, default='hubert', help="wenet or hubert")
    opt = parser.parse_args()
    asr_mode = opt.asr

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path, asr_mode)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path, asr_mode)
    
    