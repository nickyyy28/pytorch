# -*- coding = utf-8 -*-
# @Time : 2021/6/15 7:06
# @Author : 30929
# @File : demo04_transform.py
# @Software : PyCharm

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv

writer = SummaryWriter("logs")

img_path = "data/train/ants_image/6240338_93729615ec.jpg"
img = cv.imread(img_path)
tensor = transforms.ToTensor()

img_tensor = tensor(img)

writer.add_image("Tensor_img", img_tensor, 1)

writer.close()

# test....

