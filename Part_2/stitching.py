import cv2
import numpy as np
import os


def detectanddes(img1, img2):

    sift = cv2.SIFT_create()
    keyp1, des1 = sift.detectAndCompute(img1, None)
    keyp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if m[0].distance < 0.75 * m[1].distance:
            good.append(m)

    good = sorted(good, key=lambda x: x[0].distance)
    matches = np.asarray(matches)

    # orb = cv2.ORB_create()
    # keyp1, des1 = orb.detectAndCompute(img1, None)
    # keyp2, des2 = orb.detectAndCompute(img2, None)

    # matcher = cv2.BFMatcher()
    # matches = matcher.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)

    imgmatches = cv2.drawMatchesKnn(
        img1, keyp1, img2, keyp2, good[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('Results/Q4/Correspondences.png', imgmatches)

    return matches, keyp1, keyp2


def stitchimages(img1, img2):

    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    matches, kp1, kp2 = detectanddes(img1gray, img2gray)

    if len(matches[:, 0]) >= 4:
        src = np.float32(
            [kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32(
            [kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError("Not enough keypoints")

    dst = cv2.warpPerspective(
        img1, H, ((img1.shape[1] + img2.shape[1]), img2.shape[0]))

    # mask = np.zeros(img2.shape, img2.dtype)
    # center = (img2.shape[0]/2, img2.shape[1])
    # output = cv2.seamlessClone(img2, dst, mask, center, cv2.NORMAL_CLONE)

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2

    return dst


if not os.path.isdir("Results"):
    os.mkdir("Results")

imglist = []
for i in range(5):
    img = cv2.imread("images/{}.JPG".format(i+1))
    img = cv2.resize(img, (500, 500))
    imglist.append(img)


print("Stitching image 1 and 2...")
res1 = stitchimages(imglist[0], imglist[1])
cv2.imwrite("Results/Q4/Q4_1.png", res1)
print("Stitching image 3...")
res2 = stitchimages(res1, imglist[2])
cv2.imwrite("Results/Q4/Q4_2.png", res2)
print("Stitching image 4...")
res3 = stitchimages(res2, imglist[3])
cv2.imwrite("Results/Q4/Q4_3.png", res3)
print("Stitching image 5...")
res4 = stitchimages(res3, imglist[4])
cv2.imwrite("Results/Q4/Q4_4.png", res4)
