import os
import pandas as pd
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage, Resize
vipshome = r'C:\Program Files (x86)\vips-dev-8.13\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
import imagesize
from preprocess import img_preprocess


test_direc = rf"E:\Datasets\STRIP_AI\raw\test"

if __name__ == '__main__':
    info = pd.read_csv(rf"E:\Datasets\STRIP_AI\raw\test.csv")

    for idx in range(len(info)):
        patient_id = info['patient_id'][idx]
        img_num = info['image_num'][idx]

        filename = rf'{test_direc}\{patient_id}_{img_num}.tif'
        print(filename)
        Path(f"test_tif\\{patient_id}").mkdir(parents=True, exist_ok=True)

        img_preprocess(filename, f"test_tif\\{patient_id}\\{img_num}", 512)
