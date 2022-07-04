import os
import cv2
import matplotlib.pyplot as plt


def getIntensity(img):
    intensity_counts = {}

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            i = img[h, w]

            if i not in intensity_counts:
                intensity_counts[i] = 1
            else:
                intensity_counts[i] += 1

    intensity_counts = dict(
        sorted(intensity_counts.items(), key=lambda x: x[0]))

    return intensity_counts


def constrastStretching(img, counts):
    minint = list(counts.keys())[0]
    maxint = list(counts.keys())[-1]
    print(maxint, minint)

    c = (255 - 0) / (maxint - minint)
    img_s = img * c

    return img_s


def plotHist(intensity_counts, name):
    plt.figure(1)
    plt.title("Histogram")
    plt.plot(list(intensity_counts.keys()), list(intensity_counts.values()))
    plt.savefig(name)
    plt.show()


if not os.path.isdir('Results'):
    os.mkdir('Results')

img = cv2.imread("moon.tif")
cv2.imwrite('org.png', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

intensity_counts = getIntensity(img)
plotHist(intensity_counts, "Results/Q5_Histogram.png")

csimg = constrastStretching(img, intensity_counts)
cv2.imwrite('Results/Q5_StretchedImage.png', csimg)
cs_counts = getIntensity(csimg)
plotHist(cs_counts, "Results/Q5_StretchedHistogram.png")
