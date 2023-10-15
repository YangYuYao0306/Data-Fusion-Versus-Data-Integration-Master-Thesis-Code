import os
import cv2
import shutil
from ultralytics import YOLO

import torch
name = './V_HELICOPTER_056_0003.png'

# Video to Picture
# decode_video(video_name, save_dir_1, name)
# ckpt = torch.load('best.pt', map_location='cpu')
model = YOLO('./best1.pt')  # build from YAML and transfer weights

# model = YOLO('yolov8n.yaml').load('best1.pt')
# model = YOLO('yolov8n.yaml').load('best1.pt')
results = model.predict(name)
print(results)
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs