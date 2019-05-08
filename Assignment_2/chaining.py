import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
from structure_from_motion import structure_from_motion


def chaining(images):
    n_images = len(images)
    pvm = np.zeros((n_images * 2, n_images * 20))  # The Point view matrix

    orb = cv2.ORB_create(nfeatures=2000, nlevels=1, scaleFactor=1.2, patchSize=25, edgeThreshold=25)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    def cast_points(xl):
        return [(int(x.pt[0]), int(x.pt[1])) for x in xl]

    last_points, last_feats = orb.detectAndCompute(images[-1], None)

    pivot, last_right_indices = 0, {}
    for i, image in enumerate(tqdm(images)):
        points, feats = orb.detectAndCompute(image, None)
        # plot_points(image, points)

        matches = bf.match(last_feats, feats)
        new_right_indices = {}
        for m in matches:
            left_pt, right_pt = cast_points([last_points[m.queryIdx], points[m.trainIdx]])
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
    cols = (pvm > 0).sum(axis=0) > 10
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
    # plot_pvm_points(images[0], pvm)
    structure_from_motion(pvm, 3)


if __name__ == '__main__':
    main()
