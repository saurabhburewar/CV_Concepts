import cv2
import os


if not os.path.isdir("Results"):
    os.mkdir("Results")


img = cv2.imread("image.png")
mask = cv2.imread("mask.png", 0)

teleaout = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
nsout = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

cv2.imwrite("Results/inpaint1.png", teleaout)
cv2.imwrite("Results/inpaint2.png", nsout)
