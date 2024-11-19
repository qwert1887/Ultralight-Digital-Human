# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/15 22:10
import os.path

import cv2

for img_id in range(1200, 1260):
    if os.path.isfile(f"imgs/{img_id}_test.jpg"):
        continue
    # img_id = 1245
    img_path = f"../dataset/huizhang_1106/full_body_img/{img_id}.jpg"
    lms_path = f"../dataset/huizhang_1106/landmarks/{img_id}.lms"

    image = cv2.imread(img_path)

    with open(lms_path, "r") as f:
        # lines = f.readlines()
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            # print(arr)
            # 绘制每个关键点
            cv2.circle(image, (int(arr[0]), int(arr[1])), 1, (255, 0, 0), 3)
    cv2.imwrite(f"imgs/{img_id}_test.jpg", image)
