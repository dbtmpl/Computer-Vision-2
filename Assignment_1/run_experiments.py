import numpy as np
import open3d as o3d
import scipy.io
import Assignment_1.calc_IPC as IPC

# Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
transformations = np.zeros((0, 3, 4))

for i in range(99):
    file_id_source = "00000000" + "{0:0=2d}".format(i)
    file_id_target = "00000000" + "{0:0=2d}".format(i + 1)

    print(file_id_source)
    print(file_id_target)

    # Read source
    base_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_source + ".pcd")
    base_point_cloud_coords = np.asarray(base_point_cloud.points)
    base_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_source + "_normal.pcd", delimiter=' ',
                                            skip_header=11)

    # Read target
    target_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_target + ".pcd")
    target_point_cloud_coords = np.asarray(target_point_cloud.points)
    target_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_target + "_normal.pcd", delimiter=' ',
                                              skip_header=11)

    R, t = IPC.calc_IPC(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                        target_point_cloud_normal)

    # Create transformation matrix [R t] of shape 3x4
    transform = np.hstack((R, t.reshape((3, 1))))
    transformations = np.append(transformations, transform.reshape((1, 3, 4)), axis=0)

np.save("data_transformations", transformations)

# base_point_cloud = scipy.io.loadmat('Data/source.mat')["source"].T
# target_point_cloud = scipy.io.loadmat('Data/target.mat')["target"].T
