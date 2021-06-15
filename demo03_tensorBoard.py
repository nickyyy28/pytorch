# -*- coding = utf-8 -*-
# @Time : 2021/6/14 23:04
# @Author : 30929
# @File : demo03_tensorBoard.py
# @Software : PyCharm

from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import numpy as np

img_path = "data/train/ants_image/0013035.jpg"
img = cv.imread(img_path)

writer = SummaryWriter("logs")

print(img.shape)

writer.add_image("img", img, 1, dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()
'''
writer.add_image()
writer.add_scalar()


writer.close()
'''
