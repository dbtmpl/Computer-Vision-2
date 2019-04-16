import numpy as np
import open3d as o3d
import scipy.io
import Assignment_1.calc_IPC as IPC
import time


def run_experiments():
    """
    TODO: Augment function for all the experimental conditions
    :return:
    """

    start = time.time()

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 3, 4))
    transformations_direct = np.zeros((0, 3, 4))

    rotation = np.eye(3)
    translation = np.zeros(3)

    for i in range(99):
        base, target = load_point_clouds(i)

        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]

        R, t = IPC.calc_IPC(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                            target_point_cloud_normal)

        # Create transformation matrix [R t] of shape 3x4
        transform = np.hstack((R, t.reshape((3, 1))))
        transformations = np.append(transformations, transform.reshape((1, 3, 4)), axis=0)

        # Calc and save direct transformations - experiment
        rotation = R.dot(rotation)
        translation = R.dot(translation) + t
        transform_direct = np.hstack((rotation, translation.reshape((3, 1))))
        transformations_direct = np.append(transformations_direct, transform_direct.reshape((1, 3, 4)), axis=0)

    end = time.time()
    print("Time elapsed:", end - start)

    np.save("data_transformations", transformations)
    np.save("data_transformations_direct", transformations_direct)


def load_point_clouds(index, load_only_base=False):
    """

    :param index: Index of the current base point cloud
    :param load_only_base: Index of the current base point cloud
    :return: base and target point cloud data
    """
    file_id_source = "00000000" + "{0:0=2d}".format(index)
    file_id_target = "00000000" + "{0:0=2d}".format(index + 1)

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


def reconstruct_3d():
    """

    :return:
    """

    transformations = np.load("data_transformations_direct.npy")

    print(transformations.shape)

    reconstructed_data = np.zeros((0, 3))

    for i in np.arange(0, 100):
        base = load_point_clouds(i, True)
        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        A1, A1_normal = IPC.cleanInput(base_point_cloud_coords, base_point_cloud_normal)

        if i > 0:
            trans = transformations[i-1]
            inv_trans = np.linalg.inv(np.append(trans, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0))

            A1 = np.hstack((A1, np.ones((A1.shape[0], 1))))
            A1 = np.dot(A1, inv_trans.T)

        reconstructed_data = np.append(reconstructed_data, A1[:, 0:3], axis=0)

    visualize_reconstructed(reconstructed_data)


def visualize_reconstructed(reconstructed):
    """

    :param reconstructed:
    :return:
    """
    point_cloud_rec = o3d.PointCloud()
    point_cloud_rec.points = o3d.Vector3dVector(reconstructed)
    o3d.draw_geometries([point_cloud_rec])


# base_point_cloud = scipy.io.loadmat('Data/source.mat')["source"].T
# target_point_cloud = scipy.io.loadmat('Data/target.mat')["target"].T
#
# R, t = IPC.calc_IPC(base_point_cloud, target_point_cloud)


# reconstruct_3d()

# base_1 = load_point_clouds(40, True)
# # base_2 = load_point_clouds(40, True)
# #
# A1, A1_normal = IPC.cleanInput(base_1[1], base_1[2])
# # A2, A2_normal = IPC.cleanInput(base_1[1], base_1[2])
# #
# #
# visualize_reconstructed(A1)

# IPC.visualize_source_and_target(A1, A2)

# run_experiments()

reconstruct_3d()