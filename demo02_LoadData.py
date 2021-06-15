# -*- coding = utf-8 -*-
# @Time : 2021/6/14 8:56
# @Author : 30929
# @File : demo02_LoadData.py
# @Software : PyCharm

from torch.utils.data import Dataset
import numpy as np
import os
import cv2 as cv


class MyData(Dataset):
    def __init__(self, root_path, label):
        self.root_path = root_path
        self.label = label
        self.path = os.path.join(root_path, label)
        self.img_list = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.root_path, self.label, img_name)
        return cv.imread(img_path), self.label

    def __len__(self):
        return len(self.img_list)


root_path = "dataset\\train"
ant_label = "ants"
bee_label = "bees"
ant_dataset = MyData(root_path, ant_label)
bee_dataset = MyData(root_path, bee_label)

print(ant_dataset.img_list)
print(bee_dataset.img_list)

ant_img, ant_img_label = ant_dataset[0]
bee_img, bee_img_label = bee_dataset[0]

cv.imshow("ant", ant_img)
cv.imshow("bee", bee_img)

cv.waitKey(0)
