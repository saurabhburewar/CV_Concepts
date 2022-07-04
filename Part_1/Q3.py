import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def translateImg(img, shift):
    trans_matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]])
    result = np.zeros(img.shape, dtype='u1')

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            org_xy = np.array([w, h, 1])
            new_xy = np.dot(trans_matrix, org_xy)

            if 0 < new_xy[0] < img.shape[1] and 0 < new_xy[1] < img.shape[0]:
                result[new_xy[1], new_xy[0]] = img[h, w]

    return result


def scaleImg(img, factor):
    scale_matrix = np.array([[factor, 0], [0, 1]])
    result = np.zeros(img.shape, dtype='u1')

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            org_xy = np.array([w, h])
            new_xy = np.dot(scale_matrix, org_xy)

            if 0 < new_xy[0] < img.shape[1] and 0 < new_xy[1] < img.shape[0]:
                result[int(new_xy[1]), int(new_xy[0])] = img[h, w]

    return result


def rotateImg(img, degrees):
    ang = math.radians(degrees)
    rot_matrix = np.array(
        [[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    result = np.zeros(img.shape, dtype='u1')

    center = (round(((img.shape[0]+1)/2)-1), round(((img.shape[1]+1)/2)-1))

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):

            y = img.shape[0] - 1 - h - center[0]
            x = img.shape[1] - 1 - w - center[1]

            org_xy = np.array([x, y])
            new_xy = np.dot(rot_matrix, org_xy)

            new_xy[0] = center[1] - new_xy[0]
            new_xy[1] = center[0] - new_xy[1]

            if 0 <= new_xy[0] < img.shape[1] and 0 < new_xy[1] < img.shape[0]:
                result[int(new_xy[1]), int(new_xy[0]), :] = img[h, w, :]

    return result


def affine(img):
    img = translateImg(img, (50, 0))
    img = scaleImg(img, 2)
    img = rotateImg(img, 30)

    return img


def plotAll(plot1, plot2, plot3, plot4):
    figure, axis = plt.subplots(2, 2)
    figure.canvas.manager.set_window_title('Affine Transformation')

    axis[0, 0].axis("off")
    axis[0, 0].imshow(plot1)
    axis[0, 0].set_title("Translation by 2 pixels")

    axis[0, 1].axis("off")
    axis[0, 1].imshow(plot2)
    axis[0, 1].set_title("Scaling by factor of 2")

    axis[1, 0].axis("off")
    axis[1, 0].imshow(plot3)
    axis[1, 0].set_title("Rotation by 30 degrees")

    axis[1, 1].axis("off")
    axis[1, 1].imshow(plot4)
    axis[1, 1].set_title("Combination of the all three")

    plt.savefig("Results/Q3_Affine.png")
    plt.show()


if not os.path.isdir('Results'):
    os.mkdir('Results')


img = cv2.imread("barbara.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

shifted_img = translateImg(img, (2, 0))
scaled_img = scaleImg(img, 2)
rotated_img = rotateImg(img, 30)
combine_img = affine(img)

plotAll(shifted_img, scaled_img, rotated_img, combine_img)
