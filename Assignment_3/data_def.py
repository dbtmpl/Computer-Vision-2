class Mesh:
    def __init__(self, vertices, colors, triangles):
        assert triangles.shape[1] == 3
        assert vertices.shape[1] == 3
        assert colors.shape[1] == 3
        assert vertices.shape[0] == colors.shape[0]
        
        self.vertices = vertices
        self.colors = colors
        self.triangles = triangles


class PCAModel:
    def __init__(self, mean, pc, std):
        self.mean = mean
        self.pc = pc
        self.std = std
