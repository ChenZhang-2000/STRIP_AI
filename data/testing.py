from preprocess import img_resize
import os
import pandas as pd
import torch

# img_resize(fr"E:\Datasets\STRIP_AI\reduced\CE", fr"E:\Datasets\STRIP_AI\processed\CE", 512)
# img_resize(fr"E:\Datasets\STRIP_AI\reduced\LAA", fr"E:\Datasets\STRIP_AI\processed\LAA", 512)
# df = pd.DataFrame(columns=['label', 'patient_id'])
#
# idx = 0
# for label in os.listdir(fr"E:\Datasets\STRIP_AI\processed"):
#     for patient_id in os.listdir(fr"E:\Datasets\STRIP_AI\processed\{label}"):
#         df.loc[idx] = [label, patient_id]
#         idx += 1
#
# df.to_csv(fr"E:\Datasets\STRIP_AI\processed\info.csv")

# print(pd.read_csv(fr"E:\Datasets\STRIP_AI\processed\info.csv", index_col=0))
torch.zeros(3).cuda()
