#This is for data cleaning

import os
import cv2
import numpy as np
from skimage import io

def process_image(img_path):
    # Load and convert to RGB
    img = io.imread(img_path)
    img = cv2.resize(img, (32, 32))
    
    # Apply CLAHE to handle the lighting issues in BelgiumTS
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def load_dataset(base_path):
    images, labels = [], []
    for folder_num in range(62):
        path = os.path.join(base_path, format(folder_num, '05d'))
        if not os.path.exists(path): continue
        for f in os.listdir(path):
            if f.endswith(".ppm"):
                images.append(process_image(os.path.join(path, f)))
                labels.append(folder_num)
    return np.array(images) / 255.0, np.array(labels)