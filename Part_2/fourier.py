import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os


def distance(x1, x2):
    return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


def LP(shape, diameter):
    mask = np.zeros(shape[:2])
    center = (shape[0]/2, shape[1]/2)
    for x in range(shape[1]):
        for y in range(shape[0]):
            mask[y, x] = math.exp(
                ((-distance((y, x), center)**2)/(2*(diameter**2))))

    plt.imshow(mask, "gray")
    plt.title("Low pass filter")
    plt.savefig("Results/lowpass.png")
    return mask


def HP(shape, diameter):
    mask = np.zeros(shape[:2])
    center = (shape[0]/2, shape[1]/2)
    for x in range(shape[1]):
        for y in range(shape[0]):
            mask[y, x] = 1 - \
                math.exp(((-distance((y, x), center)**2)/(2*(diameter**2))))

    plt.imshow(mask, "gray")
    plt.title("High pass filter")
    plt.savefig("Results/highpass.png")
    return mask


if not os.path.isdir("Results"):
    os.mkdir("Results")

img = cv2.imread("barbara.jpg")
# img = cv2.imread("816813-chandrayaan-2.jpg.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgfft = np.fft.fftshift(np.fft.fft2(img))

lpcenter = imgfft * LP(img.shape, 10)
imglp = np.fft.ifft2(np.fft.ifftshift(lpcenter))
imglp = np.abs(imglp)

hpcenter = imgfft * HP(img.shape, 10)
imghp = np.fft.ifft2(np.fft.ifftshift(hpcenter))
imghp = np.abs(imghp)

plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(img, "gray"), plt.title("Original image")
plt.subplot(132), plt.imshow(imglp, "gray"), plt.title("Low pass filtered")
plt.subplot(133), plt.imshow(imghp, "gray"), plt.title("High pass filtered")
plt.suptitle("Low pass and High pass filters", fontweight="bold")
plt.savefig("Results/Q1_1.png")
# plt.savefig("Results/Q1_2.png")
plt.show()
