import os
import cv2
import numpy as np


def correspondenceSIFT(img1, img2):

    sift = cv2.SIFT_create()
    keyp1, des1 = sift.detectAndCompute(img1, None)
    keyp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    imgmatches = cv2.drawMatchesKnn(
        img1, keyp1, img2, keyp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('Correspondencesall.png', imgmatches)

    pointmatches = []

    for match in good[:10]:
        point = (keyp1[match[0].queryIdx].pt, keyp2[match[0].trainIdx].pt)
        pointmatches.append(point)


def normalize_coordinates(corr, K):
    normalized_corr = []
    K_inv = np.linalg.inv(K)

    for match in corr:
        new_match = []
        for point in match:
            homo_point = [point[0], point[1], 1]
            new_point = np.dot(K_inv, homo_point)
            new_match.append(new_point)

        normalized_corr.append(new_match)

    return normalized_corr


def essential(corr):
    corr = corr[:8]
    A = []

    for pair in corr:
        y = pair[0]
        x = pair[1]
        A_vec = [x[0]*y[0], x[1]*y[0], y[0], x[0]
                 * y[1], x[1]*y[1], y[1], x[0], x[1], 1]
        A.append(A_vec)

    A = np.array(A)
    U, D, V = np.linalg.svd(A)
    E_est = V[:, 8].reshape(3, 3)

    U, D, V = np.linalg.svd(E_est)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), np.transpose(V)))

    return E


def decompose_essential(E):

    U, D, V = np.linalg.svd(E)
    Rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    R = np.dot(U, np.dot(np.transpose(Rz), np.transpose(V)))
    t = np.dot(U, np.dot(Rz, np.dot(D, np.transpose(U))))

    return R, t


def get_3d_points(corr, K, R, t):

    M = K
    for row in M:
        row.append(0)

    map_mat = [[R[0][0], R[0][1], R[0][2], t[0]],
               [R[1][0], R[1][1], R[1][2], t[1]],
               [R[2][0], R[2][1], R[2][2], t[2]],
               [0, 0, 0, 1]]

    map_mat = np.array(map_mat)
    N = np.dot(M, map_mat)

    cc1 = corr[0][0]
    cc2 = corr[0][1]
    cc1 = np.array(cc1)
    cc2 = np.array(cc2)
    M = np.array(M)
    N = np.array(N)

    A1 = np.cross(cc1, M, axis=0)
    A2 = np.cross(cc2, N, axis=0)
    A = [[A1[0][0], A1[0][1], A1[0][2], A1[0][3]],
         [A1[1][0], A1[1][1], A1[1][2], A1[1][3]],
         [A1[2][0], A1[2][1], A1[2][2], A1[2][3]],
         [A2[0][0], A2[0][1], A2[0][2], A2[0][3]],
         [A2[1][0], A2[1][1], A2[1][2], A2[1][3]],
         [A2[2][0], A2[2][1], A2[2][2], A2[2][3]]]

    _, _, V = np.linalg.svd(A)
    V_est = V[:, 3]
    P_est = [V_est[0]/V_est[3], V_est[1]/V_est[3], V_est[2]/V_est[3], 1]


def correspondenceORB(img1, img2):

    orb = cv2.ORB_create()
    keyp1, des1 = orb.detectAndCompute(img1, None)
    keyp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    imgmatches = cv2.drawMatches(img1, keyp1, img2, keyp2, matches[:100], None)
    cv2.imwrite('Correspondences.png', imgmatches)

    pointmatches = []

    for match in matches:
        point = (keyp1[match.queryIdx].pt, keyp2[match.trainIdx].pt)
        pointmatches.append(point)

    return pointmatches


def main():
    if not os.path.isdir("Data"):
        print("ERROR: cannot find data")

    K = []

    with open("./Data/Intrinsic_Matrix_K.txt", 'r') as f:
        for row in f:
            krow = []
            row = row.strip().split(" ")
            for item in row:
                if item != '':
                    krow.append(float(item))

            K.append(krow)

    img1 = cv2.imread("./Data/im1.jpg")
    img2 = cv2.imread("./Data/im2.jpg")

    c_list = correspondenceORB(img1, img2)
    nor_corr = normalize_coordinates(c_list, K)
    E = essential(nor_corr)
    print("Essential matrix: \n", E)
    R, t = decompose_essential(E)
    print("Rotation matrix: \n", R)
    print("Translation vector: \n", t)
    get_3d_points(nor_corr, K, R, t)


if __name__ == "__main__":
    main()
