import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from glob import glob
from structure_from_motion import structure_from_motion
from fundamental_matrix import ransac
from argparse import ArgumentParser


AKAZE = cv2.AKAZE_create(threshold=10E-5)


def imfeature(image):
    points, features = AKAZE.detectAndCompute(image, None)
    # Convert to homogenous nparray
    points = np.hstack((cv2.KeyPoint.convert(points), np.ones((len(points), 1)))).astype(np.int)
    return points, features


def fmat_filter(matches, fmat, last_points, points, thres=0.01):
    prods = np.array([points[rid] @ fmat @ last_points[lid] for lid, rid in matches])
    return matches[prods < thres]


def spatial_thres(match_lists, last_points, points, thres=20):
    matches = []
    for match_list in match_lists:
        _match, mind = None, thres
        for match in match_list:
            eucl = np.sqrt(np.sum((last_points[match.queryIdx] - points[match.trainIdx])**2))
            if eucl < mind:
                mind = eucl
                _match = match.queryIdx, match.trainIdx
        if _match:
            matches.append(_match)
    return np.array(matches)


def chaining(images, use_ransac=False, dist_thres=20, length_thres=20):
    n_images = len(images)
    pvm = np.zeros((n_images * 2, n_images * 100))  # The Point view matrix

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    last_points, last_feats = imfeature(images[0])
    pivot, last_right_indices = 0, {}
    for i, image in enumerate(tqdm(images)):
        points, feats = imfeature(image)
        # plot_points(image, points)

        # Get nearest best matches
        match_lists = bf.knnMatch(last_feats, feats, k=4)
        matches = spatial_thres(match_lists, last_points, points, thres=dist_thres)

        # Filter with Fundamental matrix
        # Reduce to matched points
        if use_ransac:
            fmat, _, _ = ransac(100, last_points[matches[:, 0]], points[matches[:, 1]], 0.005, "norm_eight_point", False)
            matches = fmat_filter(matches, fmat, last_points, points)

        new_right_indices = {}
        for left_pt, right_pt in matches:
            idx = last_right_indices.get(left_pt, pivot)
            if left_pt not in last_right_indices:
                pivot += 1
            pvm[i*2:i*2+2, idx] = points[right_pt, :2]
            new_right_indices[right_pt] = idx
        last_right_indices = new_right_indices

        last_points, last_feats = points, feats

        if pivot > 0.9 * pvm.shape[1]:
            pvm = np.hstack((pvm, np.zeros((n_images * 2, (n_images - i) * 60))))

    # Filter PVM
    cols = (pvm > 0).sum(axis=0) > length_thres * 2
    pvm = pvm[:, cols]
    return pvm


def plot_points(image, points):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def plot_pvm(pvm):
    """
    Transforms point view matrix into binary representation and visualizes the trajectories.
    :param pvm: Previously calculated point view matrix
    """
    binary_pvm = pvm > 0
    fig, ax = plt.subplots()
    plt.imshow(binary_pvm[::2, :], cmap='binary', aspect=25)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("pvm.png", dpi=300)


def plot_pvm_points(image, pvm):
    fig, ax = plt.subplots()
    plt.imshow(image)
    for col in pvm.T:
        plt.plot(col[col > 0][::2], col[col > 0][1::2])
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig('chainz.png', dpi=300)


def plot_model(model):
    model = model[model[:, 2] > -0.25, :]  # Hacky filter, sry
    model = model[model[:, 2] < 0.5, :]  # Hacky filter, sry
    x, y, z = model[:, 0], model[:, 1], model[:, 2]
    #z *= 2  # hacky z-scaling , srysry

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='summer')


def main():
    parser = ArgumentParser()
    parser.add_argument('--ransac', action='store_true')
    parser.add_argument('--dist', type=int, default=20)
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--affinity', action='store_true')
    args = parser.parse_args()

    images = [cv2.imread(image) for image in sorted(glob("Data/House/*.png"))]
    pvm = chaining(images, use_ransac=args.ransac, dist_thres=args.dist, length_thres=args.length)
    model = structure_from_motion(pvm, args.steps, False)

    plot_pvm(pvm)
    plot_pvm_points(images[0], pvm)
    plot_model(model)
    plt.show()


if __name__ == '__main__':
    main()
