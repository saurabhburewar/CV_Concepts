import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def ltKernel(p1, p2, x):
    a = (x - p1[0]) / (p2[0] - p1[0])
    mid = (1 - a) * p1[0] + a * p2[1]

    return mid


def bcKernel(p, a):
    if (abs(p) > 0) and (abs(p) <= 1):
        return (a + 2) * (abs(p)**3) - (a + 3) * (abs(p)**2) + 1
    elif (abs(p) > 1) and (abs(p) <= 2):
        return -(a) * (abs(p)**3) - (a * 5) * (abs(p)**2) + (a * 8) * abs(p) - 4 * a
    else:
        return 0


def ltresize(img, padded):
    height, width, channels = img.shape

    new_shape = (padded.shape[0], padded.shape[1], 3)
    result = np.zeros(new_shape)


def bcResize(img, padded, a=-1/2):
    height, width, channels = img.shape

    new_shape = (int(padded.shape[0]), int(padded.shape[1]), 3)
    result = np.zeros(new_shape)

    # for h in range(int(padded.shape[0])):
    #     for w in range(int(padded.shape[1])):

    #         x1 =

    #         matrix_x = np.matrix(
    #             [[bcKernel(x1, a), bcKernel(x2, a), bcKernel(x3, a), bcKernel(x4, a)]])
    #         matrix_y = np.matrix(
    #             [[bcKernel(y1, a)], [bcKernel(y2, a)], [bcKernel(y3, a)], [bcKernel(y4, a)]])
    #         matrix_values = np.matrix([[padded[int(y-y1), int(x-x1), c],
    #                                     padded[int(y-y2), int(x-x1), c],
    #                                     padded[int(y+y3), int(x-x1), c],
    #                                     padded[int(y+y4), int(x-x1), c]],
    #                                    [padded[int(y-y1), int(x-x2), c],
    #                                     padded[int(y-y2), int(x-x2), c],
    #                                     padded[int(y+y3), int(x-x2), c],
    #                                     padded[int(y+y4), int(x-x2), c]],
    #                                    [padded[int(y-y1), int(x-x3), c],
    #                                     padded[int(y-y2), int(x-x3), c],
    #                                     padded[int(y+y3), int(x-x3), c],
    #                                     padded[int(y+y4), int(x-x3), c]],
    #                                    [padded[int(y-y1), int(x-x4), c],
    #                                     padded[int(y-y2), int(x-x4), c],
    #                                     padded[int(y+y3), int(x-x4), c],
    #                                     padded[int(y+y4), int(x-x4), c]]])

    #         mid_matrix = np.dot(matrix_x, matrix_values)
    #         result[h, w, c] = np.dot(mid_matrix, matrix_y)

    # return result


def plotCompare(plot1, plot2, plot3, name):
    if plot3 == None:
        figure, axis = plt.subplots(1, 2)
        axis[0].imshow(plot1)
        axis[0].set_title("Input")

        axis[1].imshow(plot2)
        axis[1].set_title("Output")
    else:
        figure, axis = plt.subplots(1, 3)
        axis[0].imshow(plot1)
        axis[0].set_title("Input")

        axis[1].imshow(plot2)
        axis[1].set_title("Output 1")

        axis[2].imshow(plot3)
        axis[2].set_title("Output 2")

    figure.canvas.manager.set_window_title('{} Interpolation'.format(name))

    plt.savefig("Results/Q2_{}.png".format(name))
    plt.show()


def addPadding(img, padding):
    adddimension = padding*2
    new_shape = (img.shape[0]+adddimension,
                 img.shape[1]+adddimension, img.shape[2])
    result = np.zeros(new_shape)

    result[padding:padding+img.shape[0],
           padding:padding+img.shape[1]] = img

    return result


if not os.path.isdir('Results'):
    os.mkdir('Results')

img = cv2.imread("barbara.jpg")
img = cv2.resize(img, (500, 500))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pad_img = addPadding(img, 100)
cv2.imwrite("Results/Q2_padded_img.png", pad_img)
bicubic_img = bcResize(img, pad_img)

# plotCompare(img, bicubic_img, None, "Bicubic")
