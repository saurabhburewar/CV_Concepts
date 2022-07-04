from skimage import feature
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from tqdm import tqdm


class LBP:
    def __init__(self, points, rad):
        self.points = points
        self.rad = rad

    def describe(self, img):
        lbp = feature.local_binary_pattern(
            img, self.points, self.rad, 'uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(
            0, self.points + 3), range=(0, self.points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist


feat = LBP(24, 8)
traindata = []

ftrain = open("./data/lfwpairstrain.txt", "r")
ftest = open("./data/lfwpairstest.txt", "r")
trainpairs = []
trainlabels = []
testpairs = []
testlabels = []

c = 0
for row in ftrain:
    if c == 0:
        numtrain = row
    else:
        row = row.strip().split("\t")
        if len(row) == 3:
            dire = "./data/lfw/" + row[0]
            d1 = {
                'img1':  dire + "/" + row[0] + "_" + ("%04d" % int(row[1])) + '.jpg',
                'img2':  dire + "/" + row[0] + "_" + ("%04d" % int(row[2])) + '.jpg'
            }
            trainpairs.append(d1)
            trainlabels.append(1)
        elif len(row) == 4:
            dir1 = "./data/lfw/" + row[0]
            dir2 = "./data/lfw/" + row[2]
            d1 = {
                'img1':  dir1 + "/" + row[0] + "_" + ("%04d" % int(row[1])) + '.jpg',
                'img2':  dir2 + "/" + row[2] + "_" + ("%04d" % int(row[3])) + '.jpg'
            }
            trainpairs.append(d1)
            trainlabels.append(0)
    c += 1

c = 0
for row in ftest:
    if c == 0:
        numtrain = row
    else:
        row = row.strip().split("\t")
        if len(row) == 3:
            dire = "./data/lfw/" + row[0]
            d1 = {
                'img1':  dire + "/" + row[0] + "_" + ("%04d" % int(row[1])) + '.jpg',
                'img2':  dire + "/" + row[0] + "_" + ("%04d" % int(row[2])) + '.jpg'
            }
            testpairs.append(d1)
            testlabels.append(1)
        elif len(row) == 4:
            dir1 = "./data/lfw/" + row[0]
            dir2 = "./data/lfw/" + row[2]
            d1 = {
                'img1':  dir1 + "/" + row[0] + "_" + ("%04d" % int(row[1])) + '.jpg',
                'img2':  dir2 + "/" + row[2] + "_" + ("%04d" % int(row[3])) + '.jpg'
            }
            testpairs.append(d1)
            testlabels.append(0)

    c += 1

for i in tqdm(range(len(trainpairs))):
    pair = trainpairs[i]
    img1 = cv2.imread(pair['img1'])
    img2 = cv2.imread(pair['img2'])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hist1 = feat.describe(gray1)
    hist2 = feat.describe(gray2)
    traindata.append((hist1, hist2))

traindata = np.array(traindata)
x, y, z = traindata.shape
traindata = traindata.reshape(x, y*z)
model = LinearSVC(C=100.0, random_state=42)
model.fit(traindata, trainlabels)

preds = []
corr = 0
for i in tqdm(range(len(testpairs))):
    pair = trainpairs[i]
    img1 = cv2.imread(pair['img1'])
    img2 = cv2.imread(pair['img2'])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    hist1 = feat.describe(gray1)
    hist2 = feat.describe(gray2)
    ex = [hist1, hist2]
    ex1 = np.array(ex)
    x, y = ex1.shape
    ex1 = ex1.reshape(x*y)
    prediction = model.predict(ex1.reshape(1, -1))
    preds.append(prediction[0])

count = 0
for i in range(len(testlabels)):
    if testlabels[i] == preds[i]:
        count += 1

print("Accuracy: ", count/len(testlabels))
