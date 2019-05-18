
import numpy as np
import h5py
from collections import namedtuple
import dlib
import torch
from torch import nn


class ViewportMatrix(np.ndarray):
    """Assumes the order of the coordinates to be X, Y, Z
    Where Y is the height and Z the depth.
    """

    def __new__(cls, l=-1, r=1, t=1, b=-1):
        m = np.zeros((4, 4))
        m += np.diag(((r - l) / 2, (t - b) / 2, 0.5, 1))
        m[3, :3] = ((r + l) / 2, (t + b) / 2, 0.5)
        return m.view(cls)


class PerspectiveMatrix(np.ndarray):
    """Assumes the order of the coordinates to be X, Y, Z"""
    FOV_SETTINGS = namedtuple(
        'FovSettings',
        'top bottom left right near far'
    )

    def __new__(cls, *args, **kwargs):
        return np.zeros((4, 4)).view(cls)

    def __init__(self, fov=(1, -1, 1, -1, 0.1, 10)):
        fov = self.FOV_SETTINGS(*fov)

        # Build the perspective proj matrix
        self[0, 0] = 2 * fov.near / (fov.right - fov.left)
        self[0, 2] = (fov.right + fov.left) / (fov.right - fov.left)

        self[1, 1] = 2 * fov.near / (fov.top - fov.bottom)
        self[1, 2] = (fov.top + fov.bottom) / (fov.top - fov.bottom)

        self[2, 2] = - (fov.far + fov.near) / (fov.far - fov.near)
        self[2, 3] = - 2 * fov.far * fov.near / (fov.far - fov.near)
        self[3, 2] = -1


class EnergyMin(nn.Module):

    def __init__(self, mean_shape, sigma2_shape, basis_shape, mean_expr, sigma2_expr, basis_expr):
        super().__init__()

        shape_s = sigma2_shape.shape
        shape_e = sigma2_expr.shape

        self.alpha = nn.Parameter(torch.zeros(shape_s))
        self.delta = nn.Parameter(torch.zeros(shape_e))
        T = torch.eye(4)
        T[2, 3] = -500
        self.T = nn.Parameter(T)

        self.mu_shape = torch.from_numpy(mean_shape)
        self.mu_expr = torch.from_numpy(mean_expr)
        self.sigma_shape = torch.from_numpy(sigma2_shape).sqrt()
        self.sigma_expr = torch.from_numpy(sigma2_expr).sqrt()
        self.basis_shape = torch.from_numpy(basis_shape)
        self.basis_expr = torch.from_numpy(basis_expr)

        self.V = torch.from_numpy(ViewportMatrix()).float()
        self.P = torch.from_numpy(PerspectiveMatrix()).float()



    def forward(self, p, g):

        p = torch.from_numpy(p).float()
        g = torch.from_numpy(g).float()

        # p vector where the batch is expected to be in the first axis

        p3d = self.basis_shape @ (self.alpha * self.sigma_shape) \
            + self.basis_expr @ (self.delta * self.sigma_expr)

        p3d = p + torch.cat((p3d, torch.zeros(68, 1)), dim=1)

        # Projection
        p2d = (p3d @ self.V @ self.P @ self.T)[:, :2]

        loss = torch.sum((p2d - g).norm(dim=1).pow(2))
        return loss

def rescale(points):
    m = np.max(np.abs(points)) / 2
    points = (points - np.mean(points)) / m
    return points

def main():

    # Load data
    fland = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

    # Mean shape and PCA data
    mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))[fland]
    sigma2_shape = np.asarray(bfm['shape/model/pcaVariance'])[:30]
    basis_shape = np.asarray(bfm['shape/model/pcaBasis']).reshape((-1, 3, 199))[:, :, :30][fland]

    mean_expr = np.asarray(bfm['expression/model/mean']).reshape((-1, 3))[fland]
    sigma2_expr = np.asarray(bfm['expression/model/pcaVariance'])[:20]
    basis_expr = np.asarray(bfm['expression/model/pcaBasis']).reshape((-1, 3, 100))[:, :, :20][fland]

    npoints = len(fland)

    # Instantiate the model
    model = EnergyMin(
        mean_shape,
        sigma2_shape,
        basis_shape,
        mean_expr,
        sigma2_expr,
        basis_expr
    )

    optimizer = torch.optim.Adam(model.parameters())

    points = mean_shape + mean_expr
    S = np.concatenate((points, np.ones((npoints, 1))), axis=1)

    # Loading the target image
    img = dlib.load_rgb_image("face.jpg")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)
    ground_truth = np.array([(point.x, point.y) for point in predictor(img, dets[0]).parts()])
    ground_truth = rescale(ground_truth)

    for i in range(10000):
        optimizer.zero_grad()
        out = model(S, ground_truth)
        out.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, out.item()))

    print(model.parameters())


if __name__ == "__main__":
    main()
