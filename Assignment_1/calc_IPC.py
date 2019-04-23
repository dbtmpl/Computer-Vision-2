from scipy import spatial
import numpy as np
import open3d as o3d
from pynndescent import NNDescent


def calc_icp(base_points, target_points, base_normals=None, target_normals=None, base_colors=None,
             sampling_tech="All", sample_size=None):
    """
    Performs the ICP algorithm ...
    :param base_points: point cloud to be aligned
    :param target_points: point cloud to be aligned to
    :param base_normals: normals of the base point cloud
    :param target_normals: normals of the target point cloud
    :param base_colors: Colors of the base point cloud
    :param sampling_tech: Sampling technique as string: 'all', 'uniform',
        'rnd_i' (random sub-sampling in each iteration) and 'inf_reg' (sub-sampling more from informative regions)
    :param sample_size: Sample size as int
    :return:
    """

    if base_normals is not None:
        if base_colors is not None:
            base_all, base_normals_all, base_colors = clean_input(base_points, base_normals, base_colors)
        else:
            base_all, base_normals_all = clean_input(base_points, base_normals)
        target, target_normals = clean_input(target_points, target_normals)

        # Random subsampling
        if sampling_tech == "uniform" or sampling_tech == "rnd_i":
            indices = np.random.choice(base_all.shape[0], sample_size, replace=False)
        # Subsampling from informative regions
        elif sampling_tech == "inf_reg":
            indices = sub_sampling_informative_regions(base_all, base_normals_all, base_colors, sample_size)
        # Just take em all, no samlping
        else:
            indices = range(0, len(base_all))

        base = base_all[indices]
        base_normals = base_normals_all[indices]
        print('[calc_icp] Sampling={}, #Sample={}, #Base={}, #Target={}'.format(
            sampling_tech, sample_size, base.shape[0], target.shape[0]))

    # If no normals are specified => dummy data is loaded
    else:
        base = base_points
        target = target_points

    rot = np.identity(3)
    trans = np.zeros(3)
    convergence_diff = 0.0000001
    if sampling_tech == "rnd_i":
        convergence_diff *= 1000
    # dummy values for initialization
    errors = [0.0, 200]
    index = NNDescent(target)

    # Iterate until RMS is unchanged
    # while not (np.isclose(errors[-2], errors[-1], atol=0.000001)):
    while not np.isclose(errors[-2], errors[-1], atol=10e-4):

        # 1. For each point in the base set (A1):
        match_idx, _ = index.query(base, k=1)
        matches = target[match_idx.flatten(), :]

        # Calculate current error
        errors.append(calc_rms(base, matches))
        print('\tStep: {:5d} RMS: {}'.format(len(errors) - 2, errors[-1]), end='\r')

        # 2. Refine the rotation matrix R and translation vector t using using SVD
        _r, _t = compute_svd(base, matches)

        # Updating the base point cloud
        base = base @ _r.T + _t

        # update overall R and t
        rot = _r @ rot
        trans = _r @ trans + _t

        if sampling_tech == "rnd_i":
            base_all[indices] = base
            indices = np.random.choice(base_all.shape[0], sample_size, replace=False)
            base = base_all[indices]
    print()

    # Final Visualization
    # visualize_base_and_target(base, tagret)

    # Test final transformation matrix
    # base_test, base_normal = clean_input(base, base_normals)
    # test_base = np.dot(base_test, rot.T) + trans
    # visualize_base_and_target(test_base, target)

    return rot, trans, errors[2:]


def clean_input(points, normals, colors=None):
    """
    Filters out all point where the normal is NaN and the Z-value is smaller 1.

    :param points: The raw point clouds
    :param normals: The normals corresponding to the point cloud
    :param colors: The colors of the base points
    :return: cleaned point cloud and normals
    """
    # Keep indices where row not Nan
    el_not_nan = ~np.isnan(points)
    rows_not_nan = np.logical_or(el_not_nan[:, 0], el_not_nan[:, 1], el_not_nan[:, 2])

    # remove outliers in z-direction
    zinliers = (points[:, 2] - points[:, 2].mean())/points[:, 2].std() < 1
    rows_to_keep = np.logical_and(rows_not_nan, zinliers)

    clean_points = points[rows_to_keep, :]
    clean_normals = normals[rows_to_keep, :]

    if colors is not None:
        clean_colors = colors[rows_to_keep, :]
        return clean_points, clean_normals, clean_colors

    return clean_points, clean_normals


def calc_rms(base, target):
    return np.sqrt(np.mean(np.sum(np.square(base - target))))


def compute_svd(base, target):
    """
    After: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    :param base: Base point cloud
    :param target: Target point cloud
    :return: Rotation matrix and translation vector
    """

    base_centroid = base.mean(axis=0)
    target_centroid = target.mean(axis=0)

    centered_base = base - base_centroid
    centered_target = target - target_centroid

    # W = np.eye(A1_centered.shape[0])
    # S = A1_centered.T.dot(W).dot(A2_centered)
    S = centered_base.T @ centered_target
    U, _, V = np.linalg.svd(S)

    ident = np.identity(U.shape[0])
    ident[-1, -1] = np.linalg.det(V.T @ U)

    R = V.T @ ident @ U.T
    t = target_centroid - R @ base_centroid

    return R, t


def visualize_base_and_target(base, target):
    """
    :param base: Base point cloud
    :param target: Target point cloud
    :return: Shows both point cloud in one plot
    """
    base_cloud, target_cloud = o3d.PointCloud(), o3d.PointCloud()
    base_cloud.points = o3d.Vector3dVector(base)
    target_cloud.points = o3d.Vector3dVector(target)
    o3d.draw_geometries([base_cloud, target_cloud])


def sub_sampling_informative_regions(points, normals, colors, sampling_size):
    """
    Using SIFT descriptors and Normal-Space Sampling to get a more informative sample of points

    :param points:
    :param normals:
    :param colors:
    :return:
    """

    # TODO: Decide SIFT reasonable? Might be a pain reconstructing image as x, y coordinates in point_cloud are relative

    # sift = cv.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)

    reference_vec = np.asarray([0, 0, 1])

    consine_similarities = []
    # spatial.distance.cosine apparently not broadcastable... :(
    for normal_vec in normals:
        consine_similarities.append(1 - spatial.distance.cosine(normal_vec[0:3], reference_vec))

    # Divide cosine similarities to reference vector in bins
    bins = np.arange(-1, 1, 0.1)
    inds = np.digitize(np.asarray(consine_similarities), bins)
    ind_sample = int(sampling_size / bins.shape[0])

    final_indices = np.zeros(0, dtype="int32")
    for _bin in np.arange(bins.shape[0]):
        eq_idx = np.argwhere(inds == _bin).flatten()
        if eq_idx.size is not 0:
            k = min(ind_sample, eq_idx.shape[0])
            final_indices = np.append(final_indices, np.random.choice(eq_idx, k, replace=False))

    # If some bins are emtpy, or have less entries than ind_sample, we have to fill them up to reach the sample size
    # For simplicity, we do this with plain uniform sampling => the sample will be skewed from here anyway.
    res = sampling_size - final_indices.shape[0]
    if res > 0:
        add_inx = np.random.choice(
            np.argwhere(~np.isin(np.arange(normals.shape[0]), final_indices)).flatten(), res, replace=False)
        final_indices = np.append(final_indices, add_inx)

    return final_indices
