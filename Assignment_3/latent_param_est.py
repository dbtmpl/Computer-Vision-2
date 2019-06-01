import os
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
import glob

from skimage.io import imsave

import cv2 as cv

from mpl_toolkits.mplot3d import Axes3D


# Utils
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("\t%s: %s" % (key, val))


def mesh_to_png(file_name, mesh, width=640, height=480, z_camera_translation=400):
    # mesh = trimesh.base.Trimesh(
    #     vertices=mesh.vertices,
    #     faces=mesh.triangles,
    #     vertex_colors=mesh.colors)

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, wireframe=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=np.array([1.7, 1.7, 1.7, 1.0]), bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    # Added camera translated z_camera_translation in the 0z direction w.r.t. the origin
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, z_camera_translation],
                            [0, 0, 0, 1]])

    # render scene
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)

    imsave(file_name, color)


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
        'top bottom right left near far'
    )

    def __new__(cls, *args, **kwargs):
        return np.zeros((4, 4)).view(cls)

    def __init__(self, fov=(1, -1, 1, -1, 10, 200)):
        fov = self.FOV_SETTINGS(*fov)

        # Build the perspective proj matrix
        self[0, 0] = 2 * fov.near / (fov.right - fov.left)
        self[0, 2] = (fov.right + fov.left) / (fov.right - fov.left)

        self[1, 1] = 2 * fov.near / (fov.top - fov.bottom)
        self[1, 2] = (fov.top + fov.bottom) / (fov.top - fov.bottom)

        self[2, 2] = -(- (fov.far + fov.near) / (fov.far - fov.near))
        self[2, 3] = -(- 2 * fov.far * fov.near / (fov.far - fov.near))
        self[3, 2] = -1


class EnergyMin(nn.Module):

    def __init__(self, mean_shape, sigma2_shape, basis_shape, mean_expr, sigma2_expr, basis_expr):
        super().__init__()

        shape_s = sigma2_shape.shape
        shape_e = sigma2_expr.shape

        self.alpha = nn.Parameter(torch.zeros(shape_s))
        self.delta = nn.Parameter(torch.zeros(shape_e))

        R, t = self.init_Rt()
        self.R = nn.Parameter(R)
        self.t = nn.Parameter(t)

        self.mu_shape = torch.from_numpy(mean_shape)
        self.mu_expr = torch.from_numpy(mean_expr)
        self.sigma_shape = torch.from_numpy(sigma2_shape).sqrt()
        self.sigma_expr = torch.from_numpy(sigma2_expr).sqrt()
        self.basis_shape = torch.from_numpy(basis_shape)
        self.basis_expr = torch.from_numpy(basis_expr)

        self.V = torch.from_numpy(ViewportMatrix()).float()
        self.P = torch.from_numpy(PerspectiveMatrix()).float()

    def forward(self, p, g):
        """

        :param p: 3D world coordinates of face
        :param g: 2D ground truth
        :return:
        """
        p = torch.from_numpy(p).float()
        g = torch.from_numpy(g).float()

        # x_1, y_1 = p[:, 0], p[:, 1]
        # x_2, y_2 = g[:, 0], g[:, 1]
        # plt.scatter(x_1.detach().numpy(), y_1.detach().numpy(), color="b", s=4)
        # plt.scatter(x_2.detach().numpy(), y_2.detach().numpy(), color="y", s=4)
        # plt.show()

        # p vector where the batch is expected to be in the first axis
        p3d_w = self.basis_shape @ (self.alpha * self.sigma_shape) \
                + self.basis_expr @ (self.delta * self.sigma_expr)

        p3d_w = p + torch.cat((p3d_w, torch.zeros(68, 1)), dim=1)

        p2d = self.project_points(p3d_w)

        # Plots for debugging
        # x_1, y_1 = p2d[:, 0], p2d[:, 1]
        # x_2, y_2 = g[:, 0], g[:, 1]
        # plt.scatter(x_1.detach().numpy(), y_1.detach().numpy(), color="b", s=4)
        # plt.scatter(x_2.detach().numpy(), y_2.detach().numpy(), color="y", s=4)
        # plt.show()

        loss = torch.sum((p2d - g).norm(dim=1).pow(2)) + 5 * self.alpha.pow(2).sum() + 10 * self.delta.pow(2).sum()
        return loss

    def init_Rt(self):
        R = torch.eye(3)
        # rotate 180 degrees!
        R[1, 1] = -1
        t = torch.zeros(3).view(-1, 1)
        t[1] = 1
        return R, t

    def project_points(self, p3d_w, rigid=None):
        # Projection to camera coordinates

        if rigid is None:
            T = self.construct_T()
        else:
            T = self.construct_T(rigid)

        # p3d_c = T @ torch.t(p3d_w)

        p3d_c = self.V @ self.P @ T @ torch.t(p3d_w)
        # print("BEFORE HOMOGENIZE")
        # print(p3d_c[0, :4])
        # print(p3d_c[3, :4])
        p3d_c = torch.div(p3d_c, p3d_c[3, :].view(1, -1))
        # print("AFTER HOMOGENIZE")
        # print(p3d_c[0, :4])
        # print(p3d_c[3, :4])

        # p3d_image = self.V @ p3d_c

        return torch.t(p3d_c[:2, :])
        # return torch.t(p3d_c[:2, :])

    def construct_T(self, rigid=None):
        if rigid is None:
            rigid_trans = torch.cat((self.R, self.t), dim=1)
        else:
            rigid_trans = torch.cat((rigid[0], rigid[1]), dim=1)

        # make homogeneous
        T = torch.cat((rigid_trans, torch.zeros(1, 4)), dim=0)
        T[-1, -1] = 1
        return T


def normalize_points(points, given_min_max=None):
    number_points = points.shape[0]
    norm_points = np.zeros((0, number_points))

    if given_min_max is None:
        min_max = []
        for i in np.arange(points.shape[1]):
            _min, _max = np.min(points[:, i]), np.max(points[:, i])
            norm_coords = (points[:, i] - _min) / (_max - _min)
            norm_points = np.append(norm_points, norm_coords.reshape((1, number_points)), axis=0)
            min_max.append((_min, _max))

        return norm_points.T, min_max

    else:
        for i in np.arange(points.shape[1]):
            _min, _max = given_min_max[i]
            norm_coords = (points[:, i] - _min) / (_max - _min)
            norm_points = np.append(norm_points, norm_coords.reshape((1, number_points)), axis=0)

        return norm_points.T, given_min_max


def denormalize_points(points, min_max):
    number_points = points.shape[0]
    denorm_points = np.zeros((0, number_points))

    for i in np.arange(points.shape[1]):
        _min, _max = min_max[i]
        norm_coords = points[:, i] * (_max - _min) + _min
        denorm_points = np.append(denorm_points, norm_coords.reshape((1, number_points)), axis=0)

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


def get_ground_truth_landmarks(img, predictor=None):
    if predictor is None:
        predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    ground_truth = np.array([(point.x, point.y) for point in predictor(img, dets[0]).parts()])

    return ground_truth


def exercise_4_and_5(model, optimizer, img, S_land, S_whole, face_model, triangles, number_whole_points):
    basis_shape, basis_expr = face_model

    ground_truth = get_ground_truth_landmarks(img)
    norm_ground_truth, min_max_gt = normalize_points(ground_truth)

    for i in range(300):
        optimizer.zero_grad()
        loss = model(S_land, norm_ground_truth)
        loss.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, loss.item()))

    p = torch.from_numpy(S_land).float()
    whole_p = torch.from_numpy(S_whole).float()

    # p vector where the batch is expected to be in the first axis
    p3d_w = model.basis_shape @ (model.alpha * model.sigma_shape) \
            + model.basis_expr @ (model.delta * model.sigma_expr)
    p3d_w = p + torch.cat((p3d_w, torch.zeros(68, 1)), dim=1)

    # Projection
    p2d = model.project_points(p3d_w)
    p2d = denormalize_points(p2d.detach().numpy(), min_max_gt)

    x_1, y_1 = p2d[:, 0], p2d[:, 1]
    x_2, y_2 = ground_truth[:, 0], ground_truth[:, 1]

    plt.imshow(img)
    plt.scatter(x_1, y_1, color="b", s=4)
    plt.scatter(x_2, y_2, color="y", s=4)
    plt.show()

    # All alpha and gamma applied on all 3d points_land
    whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                     + torch.from_numpy(basis_expr) @ (model.delta * model.sigma_expr)
    whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

    # projection all points_land
    whole_p2d = model.project_points(whole_p3d)
    whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
    # Exercise 5
    new_texture = find_corresponding_texture(whole_p2d, img)

    mesh = trimesh.base.Trimesh(
        vertices=whole_p3d.detach().numpy()[:, :3],
        faces=triangles,
        vertex_colors=new_texture
    )

    # color, depth = render_mesh(mesh)
    # plt.imshow(color)
    # plt.savefig("rendered_person.png")
    # plt.show()
    mesh.show()


def exercise_6(model, images, S_land, S_whole, face_model, triangles, number_whole_points):
    basis_shape, basis_expr = face_model

    gt_landmarks = []
    predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")
    for img in images:
        ground_truth = get_ground_truth_landmarks(img, predictor)
        norm_ground_truth, min_max_gt = normalize_points(ground_truth)
        gt_landmarks.append((norm_ground_truth, min_max_gt))

    params = [torch.zeros(30), torch.zeros(20)]
    for gt_landmark in gt_landmarks:
        norm_ground_truth, min_max_gt = gt_landmark

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for i in range(200):
            optimizer.zero_grad()
            out = model(S_land, norm_ground_truth)
            out.backward()
            optimizer.step()
            print("Iter: {}, loss: {}".format(i, out.item()))

        R, t = model.init_Rt()
        model.R = nn.Parameter(R)
        model.t = nn.Parameter(t)

        params[0] += model.alpha
        params[1] += model.delta
        model.alpha = nn.Parameter(torch.zeros(30))
        model.delta = nn.Parameter(torch.zeros(20))

    model.alpha = nn.Parameter(torch.mean(params[0]))
    model.delta = nn.Parameter(torch.mean(params[1]))

    p = torch.from_numpy(S_land).float()
    whole_p = torch.from_numpy(S_whole).float()

    norm_ground_truth, min_max_gt = gt_landmarks[0]
    img = images[0]

    # p vector where the batch is expected to be in the first axis
    p3d_w = model.basis_shape @ (model.alpha * model.sigma_shape) \
            + model.basis_expr @ (model.delta * model.sigma_expr)
    p3d_w = p + torch.cat((p3d_w, torch.zeros(68, 1)), dim=1)

    # Projection
    p2d = model.project_points(p3d_w)
    p2d = denormalize_points(p2d.detach().numpy(), min_max_gt)
    ground_truth = denormalize_points(norm_ground_truth, min_max_gt)

    x_1, y_1 = p2d[:, 0], p2d[:, 1]
    x_2, y_2 = ground_truth[:, 0], ground_truth[:, 1]

    plt.imshow(img)
    plt.scatter(x_1, y_1, color="b", s=4)
    plt.scatter(x_2, y_2, color="y", s=4)
    plt.show()

    # All alpha and gamma applied on all 3d points_land
    whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                     + torch.from_numpy(basis_expr) @ (model.delta * model.sigma_expr)
    whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

    # projection all points_land
    whole_p2d = model.project_points(whole_p3d)
    whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
    # Exercise 5
    new_texture = find_corresponding_texture(whole_p2d, img)

    mesh = trimesh.base.Trimesh(
        vertices=whole_p3d.detach().numpy()[:, :3],
        faces=triangles,
        vertex_colors=new_texture
    )
    color, depth = render_mesh(mesh)
    plt.imshow(color)
    plt.savefig("rendered_person.png")
    plt.show()
    mesh.show()


def exercise7(model, video_cap, S_land, S_whole, face_model, triangles, number_whole_points, min_max_w_points):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    basis_shape, basis_expr = face_model

    _ret, first_frame = video_cap.read()

    ground_truth = get_ground_truth_landmarks(first_frame)
    norm_ground_truth, min_max_gt = normalize_points(ground_truth)

    for i in range(200):
        optimizer.zero_grad()
        loss = model(S_land, norm_ground_truth)
        loss.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, loss.item()))

    model.alpha.requires_grad = False
    whole_p = torch.from_numpy(S_whole).float()

    for i in np.arange(100):
        print("Current frame: ", i)
        _ret, frame = video_cap.read()
        if _ret:
            ground_truth = get_ground_truth_landmarks(frame)
            norm_ground_truth, min_max_gt = normalize_points(ground_truth)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for j in range(200):
                optimizer.zero_grad()
                out = model(S_land, norm_ground_truth)
                out.backward()
                optimizer.step()
                print("Iter: {}, loss: {}".format(j, out.item()))

            rigid = model.R, model.t

            # All alpha and gamma applied on all 3d points_land
            whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                             + torch.from_numpy(basis_expr) @ (model.delta * model.sigma_expr)
            whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

            # projection all points_land
            whole_p2d = model.project_points(whole_p3d, rigid=rigid)
            whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
            new_texture = find_corresponding_texture(whole_p2d, frame)

            conv_p3d = denormalize_points(whole_p3d.detach().numpy()[:, :3], min_max_w_points)

            mesh = trimesh.base.Trimesh(
                vertices=conv_p3d,
                faces=triangles,
                vertex_colors=new_texture
            )

            mesh_to_png("video_results/my_mesh_{}.png".format(i), mesh)

            R, t = model.init_Rt()
            model.R = nn.Parameter(R)
            model.t = nn.Parameter(t)
            model.delta = nn.Parameter(torch.zeros(20))

        else:
            break
    video_cap.release()


def main():
    # Load data
    landmarks_model = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl", dtype=int)
    bfm = h5py.File("Data/model2017-1_face12_nomouth.h5", 'r')
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

    # Mean shape and PCA data
    mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
    mean_shape_land = mean_shape[landmarks_model]
    sigma2_shape = np.asarray(bfm['shape/model/pcaVariance'])[:30]
    basis_shape = np.asarray(bfm['shape/model/pcaBasis']).reshape((-1, 3, 199))[:, :, :30]
    basis_shape_land = basis_shape[landmarks_model]

    mean_expr = np.asarray(bfm['expression/model/mean']).reshape((-1, 3))
    mean_expr_land = mean_expr[landmarks_model]
    sigma2_expr = np.asarray(bfm['expression/model/pcaVariance'])[:20]
    basis_expr = np.asarray(bfm['expression/model/pcaBasis']).reshape((-1, 3, 100))[:, :, :20]
    basis_expr_land = basis_expr[landmarks_model]

    number_landmark = len(landmarks_model)
    number_whole_points = mean_shape.shape[0]

    # Images for exercise 4
    # img = dlib.load_rgb_image("faces/dan.jpg")
    img = dlib.load_rgb_image("faces/surprise.png")
    # img = dlib.load_rgb_image("faces/exercise_6/dave1.jpg")

    # images for exercise 6
    image_data = [cv.imread(image) for image in sorted(glob.glob("faces/exercise_6/*.jpg"))]

    # Load video for exercise 7
    video_cap = cv.VideoCapture("faces/exercise_7/smile.mp4")

    # Instantiate the model
    model = EnergyMin(
        mean_shape_land,
        sigma2_shape,
        basis_shape_land,
        mean_expr_land,
        sigma2_expr,
        basis_expr_land
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("P Matrix")
    print(model.P)

    print("V Matrix")
    print(model.V)

    # Landmark points
    points_land = mean_shape_land + mean_expr_land
    points_land, min_max_points_land = normalize_points(points_land)
    S_land = np.concatenate((points_land, np.ones((number_landmark, 1))), axis=1)

    # Total points of 3D face model
    whole_points = mean_shape + mean_expr
    whole_points_norm, min_max_w_points = normalize_points(whole_points, min_max_points_land)
    S_whole = np.concatenate((whole_points_norm, np.ones((number_whole_points, 1))), axis=1)

    face_model = basis_shape, basis_expr

    exercise_4_and_5(model, optimizer, img, S_land, S_whole, face_model, triangles, number_whole_points)
    # exercise_6(model, image_data, S_land, S_whole, face_model, triangles, number_whole_points)
    #
    # exercise7(model, video_cap, S_land, S_whole, face_model, triangles, number_whole_points, min_max_w_points)


if __name__ == "__main__":
    main()
