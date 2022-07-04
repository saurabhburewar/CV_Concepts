import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def quantization(img, k):
    bins = np.linspace(0, img.max(), k)
    result = np.digitize(img, bins)
    func = np.vectorize(bins.tolist().__getitem__)
    result = (func(result-1).astype(int))

    return result


def colorKmeans(img, k):
    img = np.array(img, dtype=np.float64) / 255
    w, h, d = img.shape
    img_2d = np.reshape(img, (w * h, d))

    sample_img = shuffle(img_2d, random_state=0)
    kmeans = KMeans(n_clusters=k, init='random', n_init=10,
                    max_iter=100, random_state=0)
    kmeans.fit(sample_img)
    labels = kmeans.predict(img_2d)
    result = kmeans.cluster_centers_[labels].reshape(w, h, -1)

    return result


def plotIQuantized(plot1, plot2, plot3, orgplot):
    figure, axis = plt.subplots(2, 2)
    figure.canvas.manager.set_window_title('K-means intensity quantization')

    axis[0, 0].axis("off")
    axis[0, 0].imshow(plot1)
    axis[0, 0].set_title("K = 2")

    axis[0, 1].axis("off")
    axis[0, 1].imshow(plot2)
    axis[0, 1].set_title("K = 4")

    axis[1, 0].axis("off")
    axis[1, 0].imshow(plot3)
    axis[1, 0].set_title("K = 8")

    axis[1, 1].axis("off")
    axis[1, 1].imshow(orgplot)
    axis[1, 1].set_title("Original image")

    plt.savefig("Results/Q1_Quantized.png")
    plt.show()


def plotCQuantized(plot1, plot2, plot3, plot4, plot5, orgplot):
    figure, axis = plt.subplots(3, 2)
    figure.canvas.manager.set_window_title('K-means color quantization')

    axis[0, 0].axis("off")
    axis[0, 0].imshow(plot1)
    axis[0, 0].set_title("K = 2")

    axis[0, 1].axis("off")
    axis[0, 1].imshow(plot2)
    axis[0, 1].set_title("K = 8")

    axis[1, 0].axis("off")
    axis[1, 0].imshow(plot3)
    axis[1, 0].set_title("K = 16")

    axis[1, 1].axis("off")
    axis[1, 1].imshow(plot4)
    axis[1, 1].set_title("K = 32")

    axis[2, 0].axis("off")
    axis[2, 0].imshow(plot5)
    axis[2, 0].set_title("K = 40")

    axis[2, 1].axis("off")
    axis[2, 1].imshow(orgplot)
    axis[2, 1].set_title("Original image")

    plt.savefig("Results/Q1_KMeansColors.png")
    plt.show()


if not os.path.isdir('Results'):
    os.mkdir('Results')

img = cv2.imread("barbara.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Intensity quantization in progress ...")
orgplot = img
plot1 = quantization(img, 2)
plot2 = quantization(img, 4)
plot3 = quantization(img, 8)
print("Plotting all images ...")
plotIQuantized(plot1, plot2, plot3, orgplot)

print("Color quantization in progress ...")
orgplot = img
plot1 = colorKmeans(img, 2)
plot2 = colorKmeans(img, 8)
plot3 = colorKmeans(img, 16)
plot4 = colorKmeans(img, 32)
plot5 = colorKmeans(img, 40)
print("Plotting all images ...")
plotCQuantized(plot1, plot2, plot3, plot4, plot5, orgplot)
