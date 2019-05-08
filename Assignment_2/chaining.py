import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
from structure_from_motion import structure_from_motion
from math import *


def chaining(images):
    n_images = len(images)
    pvm = np.zeros((n_images * 2, n_images * 40))  # The Point view matrix

    orb = cv2.ORB_create(nfeatures=1000, nlevels=2, scaleFactor=1.2, patchSize=22, edgeThreshold=22)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def spatial_thres(match_list):
        def _cast(x):
            return int(x.pt[0]), int(x.pt[1])
        mind = 10e5
        for match in match_list:
            qry, tgt = _cast(last_points[match.queryIdx]), _cast(points[match.trainIdx])
            eucl = sqrt((qry[0] - tgt[0])**2 + (qry[1] - tgt[1])**2)
            if eucl < mind:
                mind = eucl
                left, right = qry, tgt
        return left, right

    last_points, last_feats = orb.detectAndCompute(images[-1], None)
    pivot, last_right_indices = 0, {}
    for i, image in enumerate(tqdm(images)):
        points, feats = orb.detectAndCompute(image, None)
        # plot_points(image, points)

        match_lists = bf.knnMatch(last_feats, feats, k=4)
        matches = map(spatial_thres, match_lists)
        new_right_indices = {}
        for left_pt, right_pt in matches:
            idx = last_right_indices.get(left_pt, pivot)
            if left_pt not in last_right_indices:
                pivot += 1
            pvm[i*2:i*2+2, idx] = right_pt
            new_right_indices[right_pt] = idx
        last_right_indices = new_right_indices

        last_points, last_feats = points, feats

        if pivot > 0.9 * pvm.shape[1]:
            pvm = np.hstack((pvm, np.zeros((n_images * 2, (n_images - i) * 40))))

    # Filter PVM
    cols = (pvm > 0).sum(axis=0) > 20
    pvm = pvm[:, cols]
    return pvm


def plot_points(image, points):
    points = cv2.KeyPoint.convert(points)
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def plot_pvm(pvm):
    """
    Transforms point view matrix into binary representation and visualizes the trajectories.
    :param pvm: Previously calculated point view matrix
    """
    binary_pvm = pvm > 0
    plt.imshow(binary_pvm[::2, :], aspect=25)
    plt.savefig("Chaining_result.png", dpi=300)
    plt.show()


def plot_pvm_points(image, pvm):
    xy = pvm.T.flatten()
    x = xy[::2]
    y = xy[1::2]

    plt.imshow(image)
    plt.scatter(x, y)
    plt.show()


def main():
    images = [cv2.imread(image) for image in sorted(glob("Data/House/*.png"))]
    pvm = chaining(images)
    #plot_pvm(pvm)
    #plot_pvm_points(images[0], pvm)
    structure_from_motion(pvm, 3)


if __name__ == '__main__':
    main()
