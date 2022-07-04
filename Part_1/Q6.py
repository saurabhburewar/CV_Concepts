# Algorithm from this paper
# Al-Haj, Ali. (2007). Combined DWT-DCT digital image watermarking. Journal of Computer Science. 3(9). 10.3844/jcssp.2007.740.746.
# https://github.com/diptamath/DWT-DCT-Digital-Image-Watermarking


import os
import cv2
import numpy as np
import pywt
from scipy.fftpack import dct
from scipy.fftpack import idct


def extractWatermark(img):
    dwt = list(pywt.wavedec2(data=img, wavelet='haar', level=1))

    all_dct = np.empty((len(dwt[0][0]), len(dwt[0][0])))
    for i in range(0, len(dwt[0][0]), 8):
        for j in range(0, len(dwt[0][0]), 8):
            subpixels = dwt[0][i: i + 8, j: j + 8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_dct[i: i + 8, j: j + 8] = subdct

    submark = []

    for i in range(0, len(all_dct), 8):
        for j in range(0, len(all_dct), 8):
            sliced = all_dct[i: i + 8, j: j + 8]
            submark.append(sliced[5][5])

    mark = np.array(submark).reshape((128, 128)).astype('uint8')

    return mark


def watermark(img, logo):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (2048, 2048))
    logo = cv2.resize(logo, (128, 128))

    img_arr = np.array(img, dtype=np.float)
    logo_arr = np.array(img, dtype=np.float)

    dwt = list(pywt.wavedec2(data=img_arr, wavelet='haar', level=4))

    all_dct = np.empty((len(dwt[0][0]), len(dwt[0][0])))
    for i in range(0, len(dwt[0][0]), 8):
        for j in range(0, len(dwt[0][0]), 8):
            subpixels = dwt[0][i: i + 8, j: j + 8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_dct[i: i + 8, j: j + 8] = subdct

    logo_flat = logo_arr.ravel()
    i = 0
    for x in range(0, len(all_dct), 8):
        for y in range(0, len(all_dct), 8):
            if i < len(logo_flat):
                subimg = all_dct[x: x + 8, y: y + 8]
                subimg[5][5] = logo_flat[i]
                all_dct[x: x + 8, y: y + 8] = subimg
                i += 1

    all_idct = np.empty((len(all_dct[0]), len(all_dct[0])))
    for i in range(0, len(all_dct[0]), 8):
        for j in range(0, len(all_dct[0]), 8):
            subpixels = all_dct[i: i + 8, j: j + 8]
            subidct = idct(idct(subpixels.T, norm="ortho").T, norm="ortho")
            all_idct[i: i + 8, j: j + 8] = subidct

    dwt[0] = all_idct

    image_arrH = pywt.waverec2(dwt, 'haar')
    image_arrH_copy = image_arrH.clip(0, 255)
    image_arrH_copy = image_arrH_copy.astype('uint8')

    return image_arrH_copy


if not os.path.isdir('Results'):
    os.mkdir('Results')

logo = cv2.imread("Logo_IITJ.png")
img = cv2.imread("barbara.jpg")

watermarked = watermark(img, logo)
cv2.imwrite("Results/Q6_watermarked_image.png", watermarked)

extracted = extractWatermark(watermarked)
cv2.imwrite("Results/Q6_extracted_watermark.png", extracted)
