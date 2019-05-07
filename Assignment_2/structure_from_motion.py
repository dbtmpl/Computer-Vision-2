import numpy as np
import cv2 as cv
from scipy.linalg import orthogonal_procrustes
from procrustes import procrustes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d


def structure_from_motion(point_view_matrix, block_size):
    """
    
    :param point_view_matrix: 
    :param block_size: 
    :return: 
    """
    print("SFM starting")
    print("............")
    m, n = int(point_view_matrix.shape[0]), point_view_matrix.shape[2]
    # m, n = int(point_view_matrix.shape[0] / 2), point_view_matrix.shape[1]

    world_coordinates = np.zeros((n, 3))
    # point_view_matrix = point_view_matrix.reshape(m, 2, n)

    print("M, N")
    print(m, n)

    print(point_view_matrix.shape[0])
    print(block_size - 1)

    for i in np.arange(m - (block_size - 1)):
        sparse_block = point_view_matrix[i:block_size + i, :, :].reshape(2 * block_size, n)
        dense_indices = np.all(sparse_block, axis=0)
        filled_wc_indices = np.all(world_coordinates, axis=1)
        # new_indices = np.where(np.invert(filled_wc_indices[dense_indices])[0])
        new_indices = np.invert(filled_wc_indices[dense_indices])

        if np.any(new_indices):

            D = sparse_block[:, dense_indices]
            D = D - np.mean(D, axis=1).reshape(2 * block_size, 1)

            U, W, Vt = np.linalg.svd(D)
            V = Vt.T

            W = np.diag(W)[:3, :3]
            V = V[:, :3]

            S = np.sqrt(W) @ V.T
            S[0, :] = -S[0, :]
            S[1, :] = -S[1, :]

            if not np.any(filled_wc_indices):
                world_coordinates[dense_indices, :] = S.T

            else:
                X = world_coordinates[np.logical_and(filled_wc_indices, dense_indices), :]
                Y = S[:, np.invert(new_indices)].T

                d, Z, transformation = procrustes(X, Y)
                R, s, t = transformation

                _S = s * S.T @ R + t

                _transformed_points = _S[new_indices, :]
                world_coordinates[np.logical_and(dense_indices, np.invert(filled_wc_indices)), :] = _transformed_points

    world_coordinates = world_coordinates[filled_wc_indices, :]
    # world_coordinates = np.delete(world_coordinates, 200, axis=0)

    # # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = world_coordinates[:, 0]
    y = world_coordinates[:, 1]
    z = world_coordinates[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(20, 00)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.set_zlim3d(-2,0)

    plt.show()

    cloud = o3d.PointCloud()
    cloud.points = o3d.Vector3dVector(world_coordinates)
    o3d.draw_geometries([cloud], left=0, top=0)
