import os
import numpy as np
import trimesh
import pyrender

import h5py

from data_def import PCAModel, Mesh

bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

def mesh_to_png(file_name, mesh):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)

if __name__ == '__main__':
    mesh = Mesh(mean_shape, mean_tex, triangles)
    mesh_to_png("debug.png", mesh)