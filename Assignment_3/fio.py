import numpy as np

from data_def import Mesh

def load_obj(obj_path):
    vertices = []
    colors = []
    triangles = []
    with open(obj_path, 'r') as f:
        for line in f:
            tokens = line.split()
            if len(tokens) == 0:
                continue

            if tokens[0] == 'v':
                vertices.append(list(map(float, tokens[1:4])))
                if len(tokens) > 4:
                    colors.append(list(map(float, tokens[4:])))
            elif tokens[0] == 'f':
                triangles.append(
                    list(map(lambda x: int(x.split('/', 1)[0]) - 1, tokens[1:4])))
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)

    return Mesh(vertices, colors, triangles)


def save_obj(file_path, mesh):
    shp, tex, tri = mesh.vertices, mesh.colors, mesh.triangles
    with open(file_path, 'wb') as f:
        data = np.hstack((shp, tex))

        np.savetxt(f, data, fmt=' '.join(['v'] + ['%.5f'] * data.shape[1]))
        np.savetxt(f, tri + 1, fmt='f %d %d %d')
