import pywt
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def enhance(img):
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # kernel = np.ones((5, 5), np.uint8)
    cA1 = cv2.filter2D(src=cA, ddepth=-1, kernel=kernel)
    # cH1 = cH+k*pn_sequence_h
    # cV1 = cV+k*pn_sequence_v

    newcoeffs = cA1, (cH, cV, cD)
    resimg = pywt.idwt2(newcoeffs, 'haar')

    return resimg


if not os.path.isdir("Results"):
    os.mkdir("Results")

img1 = cv2.imread("normal-frontal-chest-x-ray.jpg")
img2 = cv2.imread("figure1-5e71be566aa8714a04de3386-98-left.jpeg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

resimg = enhance(img1)
# resimg = enhance(img2)
plt.imshow(resimg, cmap="gray")
plt.savefig("Results/Q2_1.png")
# plt.savefig("Q2_2.png")

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clahe.apply(cA)
