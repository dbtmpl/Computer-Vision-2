from scipy import spatial
import numpy as np
import open3d as o3d
import scipy.io
import cv2 as cv


def calc_ICP(base_point_cloud, target_point_cloud, base_point_cloud_normal=None, target_point_cloud_normal=None,
             base_point_cloud_colors=None, sampling=("all", "undf.")):
    '''
    Performs the ICP algorithm ...
    :param base_point_cloud:
    :param target_point_cloud:
    :param base_point_cloud_normal:
    :param target_point_cloud_normal:
    :param sampling: Tuple with two entries:
        (1) Sampling technique as string: 'all', 'uniform', 'rnd_i' (random sub-sampling in each iteration) and 'inf_reg' (sub-sampling more from informative regions)
        (2) Sample size as int
    :return:
    '''

    sampling_tech = sampling[0]
    sample_size = sampling[1]

    if base_point_cloud_normal is not None:
        A1_all, A1_normal_all, A1_colors_all = cleanInput(base_point_cloud, base_point_cloud_normal,
                                                          base_point_cloud_colors)
        A2_all, A2_normal_all = cleanInput(target_point_cloud, target_point_cloud_normal)

        if sampling_tech == "uniform" or sampling_tech == "rnd_i":
            indices = np.random.choice(A1_all.shape[0], sample_size, replace=False)
            A1 = A1_all[indices]
        elif sampling_tech == "inf_reg":
            indices = sub_sampling_informative_regions(A1_all, A1_normal_all, A1_colors_all, sample_size)
            A1 = A1_all[indices]
        else:  # sampling technique 'all'
            A1 = A1_all

        print(A1.shape)
        print(A2_all.shape)

    # If no normals are specified => dummy data is loaded
    else:
        A1 = base_point_cloud
        A2_all = target_point_cloud

    final_R = np.identity(3)
    final_t = np.zeros(3)

    rms_errors = []

    # dummy values for initialization
    current_rms, old_RMS = 0.0, 200.0
    # Iterate until RMS is unchanged
    while not (np.isclose(current_rms, old_RMS, atol=0.0000001)):
        old_RMS = current_rms
        # 1. For each point in the base set (A1), find with brute force the best matching point in the target point set (A2)
        matching_A2 = get_matching_targets(A1, A2_all)

        # Calculate current error
        current_rms = calc_rms(A1, matching_A2)
        print("Current RMS", current_rms)
        rms_errors.append(current_rms)

        # 2. Refine the rotation matrix R and translation vector t using using SVD
        R, t = compute_SVD(A1, matching_A2)

        # Visualization - We do this before updating to visualize the initial setting
        # visualize_source_and_target(A1, A2)

        # Updating the base point cloud
        A1 = np.dot(A1, R.T) + t

        # update overall R and t
        final_R = R.dot(final_R)
        final_t = R.dot(final_t) + t

        if sampling_tech == "rnd_i":
            indices = np.random.choice(A1_all.shape[0], sample_size, replace=False)
            A1 = A1_all[indices]
            # A1_normal = A1_normal_all[indices]

    # Final Visualization
    # visualize_source_and_target(A1, A2)

    # Test final transformation matrix
    # A1_test, A1_normal = cleanInput(base_point_cloud, base_point_cloud_normal)
    # test_A1 = np.dot(A1_test, final_R.T) + final_t
    # visualize_source_and_target(test_A1, A2)

    return final_R, final_t, rms_errors


def cleanInput(point_cloud, point_cloud_normal, point_cloud_colors=None):
    '''
    Filters out all point where the normal is NaN and the Z-value is smaller 1.

    :param point_cloud: The raw point clouds
    :param point_cloud_normal: The normals corresponding to the point cloud
    :return: cleaned point cloud and normals
    '''
    # Keep indices where row not Nan
    el_not_nan = ~np.isnan(point_cloud_normal)
    rows_not_nan = np.logical_or(el_not_nan[:, 0], el_not_nan[:, 1], el_not_nan[:, 2])

    # points shallow enough to survice (depth < 1)
    shallow_rows = point_cloud[:, 2] < 1
    rows_to_keep = np.logical_and(rows_not_nan, shallow_rows)

    A = point_cloud[rows_to_keep, :]
    A_normal = point_cloud_normal[rows_to_keep, :]

    if point_cloud_colors is not None:
        A_colors = point_cloud_colors[rows_to_keep, :]
        return A, A_normal, A_colors

    return A, A_normal


def calc_rms(A1, A2):
    return np.sqrt(np.mean(np.sum(np.square(A1 - A2))))


def get_matching_targets(base_point_cloud, target_point_cloud):
    '''

    Finds the closest point in the target point cloud for each base point cloud member.
    Note: Since the brute force approach was not feasible with our hardware, we build up a tree. This is inspired by
    the following stack overflow article: https://stackoverflow.com/questions/32446703/find-closest-vector-from-a-list-of-vectors-python

    '''

    tree = spatial.KDTree(target_point_cloud)
    min_distances, matching_indices = tree.query(base_point_cloud)
    return target_point_cloud[matching_indices, :]


def compute_SVD(A1, A2):
    """
    After: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    :param A1: Base point cloud
    :param A2: Target point cloud
    :return: Rotation matrix and translation vector
    """

    centroid_A1 = np.mean(A1, axis=0)
    centroid_A2 = np.mean(A2, axis=0)

    A1_centered = A1 - centroid_A1
    A2_centered = A2 - centroid_A2

    # W = np.eye(A1_centered.shape[0])
    # S = A1_centered.T.dot(W).dot(A2_centered)
    S = A1_centered.T.dot(A2_centered)
    U, _, V = np.linalg.svd(S)

    ident = np.identity(U.shape[0])
    ident[-1, -1] = np.linalg.det(V.T.dot(U))

    R = V.T.dot(ident).dot(U.T)
    t = centroid_A2 - R.dot(centroid_A1)

    return R, t


def visualize_source_and_target(A1, A2):
    '''

    :param A1: Base point cloud
    :param A2: Target point cloud
    :return: Shows both point cloud in one plot
    '''
    point_cloud_source = o3d.PointCloud()
    point_cloud_target = o3d.PointCloud()
    point_cloud_source.points = o3d.Vector3dVector(A1)
    point_cloud_target.points = o3d.Vector3dVector(A2)
    o3d.draw_geometries([point_cloud_source, point_cloud_target])


def sub_sampling_informative_regions(point_cloud, point_cloud_normals, point_cloud_colors, sampling_size):
    """
    Using SIFT descriptors and Normal-Space Sampling to get a more informative sample of points

    :param point_cloud:
    :param point_cloud_normals:
    :param point_cloud_colors:
    :return:
    """

    # TODO: Decide SIFT reasonable? Might be a pain reconstructing image as x, y coordinates in point_cloud are relative

    # sift = cv.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)

    reference_vec = np.asarray([0, 0, 1])

    consine_similarities = []
    # spatial.distance.cosine apparently not broadcastable... :(
    for normal_vec in point_cloud_normals:
        consine_similarities.append(1 - spatial.distance.cosine(normal_vec[0:3], reference_vec))

    # Divide cosine similarities to reference vector in bins
    bins = np.arange(-1, 1, 0.1)
    inds = np.digitize(np.asarray(consine_similarities), bins)
    ind_sample = int(sampling_size / bins.shape[0])

    final_indices = np.zeros(0, dtype="int32")
    for bin in np.arange(bins.shape[0]):
        eq_idx = np.argwhere(inds == bin).flatten()
        if eq_idx.size is not 0:
            k = min(ind_sample, eq_idx.shape[0])
            final_indices = np.append(final_indices, np.random.choice(eq_idx, k, replace=False))

    # If some bins are emtpy, or have less entries than ind_sample, we have to fill them up to reach the sample size
    # For simplicity, we do this with plain uniform sampling => the sample will be skewed from here anyway.
    res = sampling_size - final_indices.shape[0]
    if res > 0:
        add_inx = np.random.choice(
            np.argwhere(~np.isin(np.arange(point_cloud_normals.shape[0]), final_indices)).flatten(), res, replace=False)
        final_indices = np.append(final_indices, add_inx)

    return final_indices
