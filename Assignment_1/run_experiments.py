import numpy as np
import open3d as o3d
from calc_IPC import calc_icp, clean_input
import time


def estimate_transformations(sample_size, sample_technique):
    """
    TODO: Augment function for all the experimental conditions
    :return:
    """

    start_time = time.time()

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 4, 4))

    rotation = np.eye(3)
    translation = np.zeros(3)

    for i in np.arange(1):
        base, target = load_point_clouds(i)

        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]

        base_point_cloud_colors = np.asarray(base[0].colors)

        R, t = calc_icp(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                        target_point_cloud_normal, base_point_cloud_colors, sample_technique, sample_size)

        # Calc and save direct transformations - experiment
        rotation = R.dot(rotation)
        translation = R.dot(translation) + t
        transform = np.hstack((rotation, translation.reshape((3, 1))))
        transform_affine = np.append(transform, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0)
        transformations = np.append(transformations, transform_affine.reshape((1, 4, 4)), axis=0)

    end_time = time.time()
    print("Time elapsed:", end_time - start_time)

    np.save("Transformations/data_transformations_sample_" + str(sample_size) + "_" + sample_technique, transformations)


def load_point_clouds(index, load_only_base=False):
    """

    :param index: Index of the current base point cloud
    :param load_only_base: Index of the current base point cloud
    :return: base and target point cloud data
    """
    file_id_source = "00000000" + "{0:0=2d}".format(index + 1)
    file_id_target = "00000000" + "{0:0=2d}".format(index)

    print(file_id_source)
    # print(file_id_target)

    # Read source
    base_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_source + ".pcd")
    base_point_cloud_coords = np.asarray(base_point_cloud.points)
    base_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_source + "_normal.pcd", delimiter=' ',
                                            skip_header=11)

    if load_only_base:
        return base_point_cloud, base_point_cloud_coords, base_point_cloud_normal

    # Read target
    target_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_target + ".pcd")
    target_point_cloud_coords = np.asarray(target_point_cloud.points)
    target_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_target + "_normal.pcd", delimiter=' ',
                                              skip_header=11)

    base = (base_point_cloud, base_point_cloud_coords, base_point_cloud_normal)
    target = (target_point_cloud, target_point_cloud_coords, target_point_cloud_normal)

    return base, target


def reconstruct_3d(sample_size, sample_technique):
    """

    :return:
    """

    transformations = np.load(
        "Transformations/data_transformations_sample_" + str(sample_size) + "_" + sample_technique + ".npy")
    # transformations = np.load("Transformations/data_transformations_sample_5000_uniform.npy")
    print(transformations.shape)

    reconstructed_data = np.zeros((0, 3))

    for i in np.arange(99):
        # for i in np.arange(20)[::-1]:
        # for i in np.concatenate((np.arange(30), np.arange(60, 100))):
        base = load_point_clouds(i, True)
        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        A1, A1_normal = clean_input(base_point_cloud_coords, base_point_cloud_normal)

        if i > 0:
            trans = transformations[i - 1]

            A1 = np.hstack((A1, np.ones((A1.shape[0], 1))))
            A1 = np.dot(A1, trans.T)

        reconstructed_data = np.append(reconstructed_data, A1[:, 0:3], axis=0)

    visualize_reconstructed(reconstructed_data)


# Test function for debugging. Alters transformations that they map on 99 and not on 0
# Can most likely be deleted
def rearange_transformation_order(transformations):
    n = transformations.shape[0]

    transformations_new = np.zeros((0, 3, 4))

    for i in np.arange(100)[::-1]:
        rotation = np.eye(3)
        translation = np.zeros(3)
        for j in np.arange(i, n):
            trans = transformations[j]
            R = trans[:, 0:3]
            t = trans[:, 3]

            rotation = R.dot(rotation)
            translation = R.dot(translation) + t

        trans_new = np.hstack((rotation, translation.reshape((3, 1))))
        transformations_new = np.append(transformations_new, trans_new.reshape((1, 3, 4)), axis=0)

    np.save("Transformations/transformations_rewind", np.flip(transformations_new, 0))


def visualize_reconstructed(reconstructed):
    """

    :param reconstructed:
    :return:
    """
    point_cloud_rec = o3d.PointCloud()
    point_cloud_rec.points = o3d.Vector3dVector(reconstructed)
    o3d.draw_geometries([point_cloud_rec])


def run_experiments_ex_2():
    """
    Test ICP algorithm w.r.t. (a) accuracy, (b) speed, (c) stability and (d) tolerance to noise by changing the
    point selection technique: (a) all the points, (b) uniform sub-sampling, (c) random sub-sampling in each iteration
    and (d) sub-sampling more from informative regions
    :return:
    """

    # TODO: Run all experiments for exercise 2.1

    pass


def run_experiments_ex_3():
    """
    - 3.1 (b) camera pose and merge the results using every 2 nd , 4 th , and 10 th frames
    - 3.2 ...
    :return:
    """
    # TODO: Run experiments for excericise 3.X

    pass


# base_point_cloud = scipy.io.loadmat('Data/source.mat')["source"].T
# target_point_cloud = scipy.io.loadmat('Data/target.mat')["target"].T

# R, t = IPC.calc_IPC(base_point_cloud, target_point_cloud)


estimate_transformations(5000, "inf_reg")
# reconstruct_3d(5000, "uniform")

# base = load_point_clouds(20, True)
