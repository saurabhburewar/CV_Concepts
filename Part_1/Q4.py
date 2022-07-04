import os
import cv2
import matplotlib.pyplot as plt


def GaussianBlur(img):
    imgsize = (img.shape[0], img.shape[1])
    copy = img.copy()
    gp = [copy]

    for i in range(3):
        copy = cv2.pyrDown(copy)
        gp.append(copy)

    blurred = cv2.pyrUp(copy)
    blurred = cv2.resize(blurred, imgsize)
    cv2.imwrite("Results/Q4_gaussianBlur.png", blurred)

    return gp


def LaplacianPy(gp):
    lp = [gp[2]]

    for i in range(2, 0, -1):
        gextended = cv2.pyrUp(gp[i], dstsize=(
            gp[i-1].shape[1], gp[i-1].shape[0]))
        l = cv2.subtract(gp[i-1], gextended)
        lp.append(l)

    cv2.imwrite("Results/Q4_laplacianFilter.png", l)

    return lp


def plotGPyramid(pyramid, name):
    figure, axis = plt.subplots(2, 2)
    figure.canvas.manager.set_window_title('Plotting pyramids')

    axis[0, 0].imshow(pyramid[0])
    axis[0, 0].set_title("Original image")

    axis[0, 1].imshow(pyramid[1])
    axis[0, 1].set_title("Level 1")

    axis[1, 0].imshow(pyramid[2])
    axis[1, 0].set_title("Level 2")

    axis[1, 1].imshow(pyramid[3])
    axis[1, 1].set_title("Level 3")

    plt.savefig("Results/Q4_{}.png".format(name))
    plt.show()


def plotLPyramid(pyramid, name):
    figure, axis = plt.subplots(1, 3)
    figure.canvas.manager.set_window_title('Plotting pyramids')

    axis[0].imshow(pyramid[0])
    axis[0].set_title("GP image")

    axis[1].imshow(pyramid[2])
    axis[1].set_title("Level 1")

    axis[2].imshow(pyramid[1])
    axis[2].set_title("Level 2")

    plt.savefig("Results/Q4_{}.png".format(name))
    plt.show()


if not os.path.isdir('Results'):
    os.mkdir('Results')

img = cv2.imread("barbara.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Calulating Gaussian pyramid...")
gp = GaussianBlur(img)
print("Calulating Laplacian pyramid...")
lp = LaplacianPy(gp)

plotGPyramid(gp, "GaussianPyramid")
plotLPyramid(lp, "LaplacianPyramid")
