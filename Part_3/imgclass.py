from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(256, 256)
        self.out = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hid(x)
        x = self.sigmoid(x)
        x = self.out(x)
        x = self.softmax(x)
        return x


with open('data/train/data_batch_1', 'rb') as f:
    data = pickle.load(f)

traindata = []
trainlabels = []

img = imread('data/barbara.jpg')
resized = resize(img, (128, 64))
fd, hogimg = hog(resized, orientations=9, pixels_per_cell=(
    8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)


def gethog():
    for image in data:
        resized = resize(image, (128, 64))
        fd, hogimg = hog(resized, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

        traindata.append(fd)


def getsift():
    for image in data:
        resized = resize(image, (128, 64))
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(resized)

        traindata.append(kp)


gethog()
# getsift()
traindata = np.array(traindata)
trainlabels = np.array(trainlabels)

model = Net()
