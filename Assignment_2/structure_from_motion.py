import numpy as np
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
    m, n = int(point_view_matrix.shape[0] / 2), point_view_matrix.shape[1]
    print("SFM starting\n.........\n#Images: {}, #Points: {}".format(m, n))

    model = np.zeros((n, 3))
    for i in range(m - (block_size - 1)):
        pvm_rows = point_view_matrix[i:i+2*block_size, ...]
        dense_idx = np.all(pvm_rows, axis=0)
        world_idx = np.all(model, axis=1)

        if np.any(~world_idx[dense_idx]):
            D = pvm_rows[:, dense_idx]
            D = D - D.mean(axis=1)[:, None]

            _, W, Vt = np.linalg.svd(D)
            V = Vt.T

            W = np.diag(W)[:3, :3]
            V = V[:, :3]

            S = np.sqrt(W) @ V.T
            S[0, :] = -S[0, :]
            S[1, :] = -S[1, :]

            if not np.any(world_idx):
                model[dense_idx, :] = S.T

            else:
                X = model[world_idx & dense_idx, :]
                Y = S[:, world_idx[dense_idx]].T
                _, _, (R, s, t) = procrustes(X, Y)
                Z = s * S.T @ R + t
                model[dense_idx, :] = Z

    model = model[world_idx, :]
    model = model[model[:, 2] > -1, :]  # Hacky filter, sry

    # # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = model[:, 0], model[:, 1],model[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(20, 00)
    plt.show()

    cloud = o3d.PointCloud()
    cloud.points = o3d.Vector3dVector(model)
    o3d.draw_geometries([cloud])


if __name__ == '__main__':
    pvm = np.loadtxt('PointViewMatrix.txt')
    structure_from_motion(pvm, 3)

