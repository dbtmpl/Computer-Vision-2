import numpy as np
import h5py
from collections import namedtuple
import dlib
import torch
from torch import nn
import matplotlib.pyplot as plt
import trimesh
from trimesh.viewer import SceneViewer
import pyrender

from mpl_toolkits.mplot3d import Axes3D


# Utils
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("\t%s: %s" % (key, val))


def render_mesh(mesh):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 250
    scene.add(camera, pose=camera_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=0.01)
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)
    return color, depth


def get_mesh(shape):
    mesh = trimesh.base.Trimesh(vertices=shape, faces=triangles, vertex_colors=mean_tex)
    return mesh


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
        # does it help to rotate 180 degrees?
        # T[1, 1] = -1
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

        loss = torch.sum((p2d - g).norm(dim=1).pow(2)) \
               + 0.5 * self.alpha.pow(2).sum() \
               + 0.5 * self.delta.pow(2).sum()
        return loss


def normalize_points(points, given_min_max=None):
    npoints = points.shape[0]
    norm_points = np.zeros((0, npoints))

    if given_min_max is None:
        min_max = []
        for i in np.arange(points.shape[1]):
            _min, _max = np.min(points[:, i]), np.max(points[:, i])
            norm_coords = (points[:, i] - _min) / (_max - _min)
            norm_points = np.append(norm_points, norm_coords.reshape((1, npoints)), axis=0)
            min_max.append((_min, _max))

        return norm_points.T, min_max

    else:
        for i in np.arange(points.shape[1]):
            _min, _max = given_min_max[i]
            norm_coords = (points[:, i] - _min) / (_max - _min)
            norm_points = np.append(norm_points, norm_coords.reshape((1, npoints)), axis=0)

        return norm_points.T, given_min_max


def denormalize_points(points, min_max):
    npoints = points.shape[0]
    denorm_points = np.zeros((0, npoints))

    for i in np.arange(points.shape[1]):
        _min, _max = min_max[i]
        norm_coords = points[:, i] * (_max - _min) + _min
        denorm_points = np.append(denorm_points, norm_coords.reshape((1, npoints)), axis=0)

    return denorm_points.T


def find_corresponding_texture(points, image):
    im_height, im_width, _ = image.shape

    new_texture = np.zeros((0, 3))
    out_of_bounds = []

    for i, point in enumerate(points):
        x, y = point[0] + 1e-2, point[1] + 1e-2
        if y < 0 or y > im_height or x < 0 or x > im_width:
            # Save indices of points that are outside the image, such that we may delete those later
            out_of_bounds.append(i)
            continue
        x_low, x_high = int(np.floor(x)), int(np.ceil(x))
        y_low, y_high = int(np.floor(y)), int(np.ceil(y))
        p_x, p_y = (x - x_low) / (x_high - x_low), (y - y_low) / (y_high - y_low)

        color_11, color_12, color_21, color_22 = image[y_low, x_low], image[y_low, x_high], \
                                                 image[y_high, x_low], image[y_high, x_high]

        hori_color_1, hori_color_2 = color_11 * (1 - p_x) + color_12 * p_x, color_21 * (1 - p_x) + color_22 * p_x
        final_color = hori_color_1 * (1 - p_y) + hori_color_2 * p_y

        new_texture = np.append(new_texture, final_color.reshape((1, 3)), axis=0)

    return new_texture


def main():
    # Load data
    fland = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)
    bfm = h5py.File("Data/model2017-1_face12_nomouth.h5", 'r')
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

    # Mean shape and PCA data
    mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
    mean_shape_fland = mean_shape[fland]
    sigma2_shape = np.asarray(bfm['shape/model/pcaVariance'])[:30]
    basis_shape = np.asarray(bfm['shape/model/pcaBasis']).reshape((-1, 3, 199))[:, :, :30]
    basis_shape_fland = basis_shape[fland]

    mean_expr = np.asarray(bfm['expression/model/mean']).reshape((-1, 3))
    mean_expr_fland = mean_expr[fland]
    sigma2_expr = np.asarray(bfm['expression/model/pcaVariance'])[:20]
    basis_expr = np.asarray(bfm['expression/model/pcaBasis']).reshape((-1, 3, 100))[:, :, :20]
    basis_expr_fland = basis_expr[fland]

    npoints = len(fland)
    n_whole_points = mean_shape.shape[0]

    # Instantiate the model
    model = EnergyMin(
        mean_shape_fland,
        sigma2_shape,
        basis_shape_fland,
        mean_expr_fland,
        sigma2_expr,
        basis_expr_fland
    )

    print("P Matrix")
    print(model.P)

    print("V Matrix")
    print(model.V)

    optimizer = torch.optim.Adam(model.parameters())

    points = mean_shape_fland + mean_expr_fland
    points, min_max_points = normalize_points(points)
    S = np.concatenate((points, np.ones((npoints, 1))), axis=1)

    whole_points = mean_shape + mean_expr
    whole_points_norm, min_max_w_points = normalize_points(whole_points, min_max_points)
    whole_S = np.concatenate((whole_points_norm, np.ones((n_whole_points, 1))), axis=1)

    whole_S_plot = np.concatenate((whole_points, np.ones((n_whole_points, 1))), axis=1)

    # Loading the target image
    img = dlib.load_rgb_image("faces/dan.jpg")
    # img = dlib.load_rgb_image("faces/surprise.png")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)
    ground_truth = np.array([(point.x, point.y) for point in predictor(img, dets[0]).parts()])
    norm_ground_truth, min_max_gt = normalize_points(ground_truth)

    for i in range(10000):
        optimizer.zero_grad()
        out = model(S, norm_ground_truth)
        out.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, out.item()))

    p = torch.from_numpy(S).float()
    whole_p = torch.from_numpy(whole_S).float()

    whole_p_plot = torch.from_numpy(whole_S_plot).float()

    # p vector where the batch is expected to be in the first axis
    p3d = model.basis_shape @ (model.alpha * model.sigma_shape) \
          + model.basis_expr @ (model.delta * model.sigma_expr)
    p3d = p + torch.cat((p3d, torch.zeros(68, 1)), dim=1)

    # All alpha and gamma applied on all 3d points
    whole_p3d_base = torch.from_numpy(basis_shape).float() @ (model.alpha * model.sigma_shape) \
                     + torch.from_numpy(basis_expr).float() @ (model.delta * model.sigma_expr)
    whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(n_whole_points, 1)), dim=1)

    whole_p3d_plot = whole_p_plot + torch.cat((whole_p3d_base, torch.zeros(n_whole_points, 1)), dim=1)

    # Projection
    p2d = (p3d @ model.V @ model.P @ model.T)[:, :2]
    p2d = denormalize_points(p2d.detach().numpy(), min_max_gt)

    # projection all points
    whole_p2d = (whole_p3d @ model.V @ model.P @ model.T)[:, :2]
    whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
    # Exercise 5
    new_texture = find_corresponding_texture(whole_p2d, img)

    # whole_p3d = (whole_p3d @ model.V @ model.P @ model.T)[:, :3]
    # whole_p3d = denormalize_points(whole_p3d.detach().numpy()[:, :3], min_max_w_points)

    x_1, y_1 = p2d[:, 0], p2d[:, 1]
    x_2, y_2 = ground_truth[:, 0], ground_truth[:, 1]

    plt.imshow(img)
    plt.scatter(x_1, y_1, color="b", s=3)
    plt.scatter(x_2, y_2, color="r", s=3)
    plt.show()

    # x = whole_p3d[:, 0].detach().numpy()
    # y = whole_p3d[:, 1].detach().numpy()
    # z = whole_p3d[:, 2].detach().numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(x, y, z, c=z, cmap='summer')
    # plt.show()

    mesh = trimesh.base.Trimesh(
        vertices=whole_p3d_plot.detach().numpy()[:, :3],
        faces=triangles,
        vertex_colors=new_texture
    )
    # color, depth = render_mesh(mesh)
    # plt.imshow(color)
    mesh.show()

    # plt.savefig("sampled_faces.png")
    # plt.show()

    # TODO: estimate mapping that it aligns with image
    # TODO: Then exercise 5 should be quite straight forward


if __name__ == "__main__":
    main()
