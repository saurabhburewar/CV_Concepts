import cv2
import numpy as np
from tqdm import tqdm


class Superpixel():
    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b


def cal_grad(h, w, img, height, width):
    if w + 1 >= width:
        w = width - 2
    if h + 1 >= height:
        h = height - 2

    return int(img[w + 1, h + 1][0]) - int(img[w, h][0]) + int(img[w + 1, h + 1][1]) - int(img[w, h][1]) + int(img[w + 1, h + 1][2]) - int(img[w, h][2])


def slic(iterations, S, img, height, width):

    cluster_list = []
    assignments = {}
    dist = np.full((img.shape[0], img.shape[1]), np.inf)

    # Set up initial cluster centers
    for h in range(S//2, height, S):
        for w in range(S//2, width, S):
            cluster_list.append(Superpixel(
                h, w, img[h, w][0], img[h, w][1], img[h, w][2]))
        w = S//2

    # Reassign cluster centers to lowest gradient pixel
    for cluster in cluster_list:
        cluster_grad = cal_grad(cluster.h, cluster.w, img, height, width)
        for someh in range(-1, 2):
            for somew in range(-1, 2):
                H = cluster.h + someh
                W = cluster.w + somew
                new_grad = cal_grad(H, W, img, height, width)
                if new_grad < cluster_grad:
                    cluster.update(H, W, img[H, W][0],
                                   img[H, W][1], img[H, W][2])
                    cluster_grad = new_grad

    # Run the algorithm for given number of iterations
    for i in tqdm(range(iterations)):

        # Assign pixels to a cluster
        for cluster in cluster_list:
            for someh in range(cluster.h - 2 * S, cluster.h + 2 * S):
                if someh < 0 or someh >= height:
                    continue
                for somew in range(cluster.w - 2 * S, cluster.w + 2 * S):
                    if somew < 0 or somew >= height:
                        continue
                    Dc = ((int(img[someh, somew][0]) - int(cluster.l))**2 + (int(img[someh, somew][1]) - int(
                        cluster.a))**2 + (int(img[someh, somew][2]) - int(cluster.b))**2)**0.5
                    Ds = ((someh - cluster.h)**2 + (somew - cluster.w)**2)**0.5
                    D = ((Dc / 20)**2 + (Ds / S)**2)**0.5
                    if D < dist[someh, somew]:
                        if (someh, somew) not in assignments:
                            assignments[(someh, somew)] = cluster
                            cluster.pixels.append((someh, somew))
                        else:
                            assignments[(someh, somew)].pixels.remove(
                                (someh, somew))
                            assignments[(someh, somew)] = cluster
                            cluster.pixels.append((someh, somew))
                        dist[someh, somew] = D

        # Update cluster mean
        for cluster in cluster_list:
            height_sum = 0
            width_sum = 0
            num = 0
            for pixel in cluster.pixels:
                height_sum += pixel[0]
                width_sum += pixel[1]
                num += 1
                H = height_sum // num
                W = width_sum // num
                cluster.update(H, W, img[H, W][0], img[H, W][1], img[H, W][2])

    return cluster_list


Size = 400
k = 100
iterations = 10

img = cv2.imread("Q1.png")
img = cv2.resize(img, (Size, Size))
cv2.imwrite("Original.png", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

N = img.shape[0] * img.shape[1]
S = (N / k)**0.5
S = int(S)

cluster_list = slic(iterations, S, img, img.shape[0], img.shape[1])

img_copy = np.copy(img)

for cluster in cluster_list:
    for pixel in cluster.pixels:
        img_copy[pixel[0], pixel[1]][0] = cluster.l
        img_copy[pixel[0], pixel[1]][1] = cluster.a
        img_copy[pixel[0], pixel[1]][2] = cluster.b

    img_copy[cluster.h, cluster.w][0] = 0
    img_copy[cluster.h, cluster.w][1] = 0
    img_copy[cluster.h, cluster.w][2] = 0

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_LAB2BGR)
cv2.imwrite("Ans1.png", img_copy)
