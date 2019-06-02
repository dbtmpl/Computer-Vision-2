import numpy as np
import h5py
from collections import namedtuple
import dlib
import torch
from torch import nn
import matplotlib.pyplot as plt
import trimesh
import pyrender
import cv2 as cv

from skimage.io import imsave
from mpl_toolkits.mplot3d import Axes3D


# Utils
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("\t%s: %s" % (key, val))


def mesh_to_png(file_name, mesh, width=640, height=480, z_camera_translation=275):
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

    def __new__(cls, l, b, t, r):
        m = np.zeros((4, 4))
        m += np.diag(((r - l) / 2, -(t - b) / 2, 0.5, 1))
        m[:3, 3] = ((r + l) / 2, (t + b) / 2, 0.5)
        return m.view(cls)


class PerspectiveMatrix(np.ndarray):
    """Assumes the order of the coordinates to be X, Y, Z"""
    FOV_SETTINGS = namedtuple(
        'FovSettings',
        'aspect_ratio near far'
    )

    def __new__(cls, *args, **kwargs):
        return np.zeros((4, 4)).view(cls)

    def __init__(self, fov=(1, 300, 2000)):
        super(PerspectiveMatrix, self).__init__()
        fov = self.FOV_SETTINGS(*fov)

        fovy = 1
        top = np.tan(fovy / 2) * fov.near
        bottom = -top
        right = top * fov.aspect_ratio
        left = -right

        # Build the perspective proj matrix
        self[0, 0] = 2 * fov.near / (right - left)
        self[0, 2] = (right + left) / (right - left)

        self[1, 1] = 2 * fov.near / (top - bottom)
        self[1, 2] = (top + bottom) / (top - bottom)

        self[2, 2] = - (fov.far + fov.near) / (fov.far - fov.near)
        self[2, 3] = - 2 * fov.far * fov.near / (fov.far - fov.near)
        self[3, 2] = -1


class EnergyMin(nn.Module):

    def __init__(self, mean_shape, sigma2_shape, basis_shape, mean_expr, sigma2_expr, basis_expr, im_shape):
        super().__init__()

        height, width, _ = im_shape
        aspect_ratio = width / height

        shape_s = sigma2_shape.shape
        shape_e = sigma2_expr.shape

        self.alpha = nn.Parameter(torch.zeros(shape_s))
        self.delta = nn.Parameter(torch.zeros((1,) + shape_e))

        self.init_Rt()

        self.mu_shape = torch.from_numpy(mean_shape)
        self.mu_expr = torch.from_numpy(mean_expr)
        self.sigma_shape = torch.from_numpy(sigma2_shape).sqrt()
        self.sigma_expr = torch.from_numpy(sigma2_expr).sqrt()
        self.basis_shape = torch.from_numpy(basis_shape)
        self.basis_expr = torch.from_numpy(basis_expr)

        # self.P = torch.from_numpy(PerspectiveMatrix((0, 0, bottom, right, 10, 2000))).float()
        self.P = torch.from_numpy(PerspectiveMatrix((aspect_ratio, 300, 2000))).float()
        self.V = torch.from_numpy(ViewportMatrix(0, 0, height, width)).float()

    def forward(self, p, g):
        """

        :param p: 3D world coordinates of face
        :param g: 2D ground truth
        :return:
        """
        p = torch.from_numpy(p).float()
        g = torch.from_numpy(g).float()

        bs = g.shape[0] if g.dim() == 3 else 1
        assert self.delta.shape == (bs, self.delta.shape[-1])

        # x_1, y_1 = p[:, 0], p[:, 1]
        # x_2, y_2 = g[:, 0], g[:, 1]
        # plt.scatter(x_1.detach().numpy(), y_1.detach().numpy(), color="b", s=4)
        # plt.scatter(x_2.detach().numpy(), y_2.detach().numpy(), color="y", s=4)
        # plt.show()
        p3d_shape = self.basis_shape @ (self.alpha * self.sigma_shape)
        p3d_expr = (self.basis_expr @ (self.delta * self.sigma_expr).t()).permute((2, 0, 1))
        p3d_w = p + torch.cat((p3d_shape + p3d_expr, torch.zeros(bs, 68, 1)), dim=-1)

        p2d = torch.zeros((bs, 68, 2))
        for i in range(bs):
            p2d[i, ...] = self.project_points(p3d_w[i, ...])

        # Plots for debugging
        # x_1, y_1 = p2d[:, 0], p2d[:, 1]
        # x_2, y_2 = g[:, 0], g[:, 1]
        # plt.scatter(x_1.detach().numpy(), y_1.detach().numpy(), color="b", s=4)
        # plt.scatter(x_2.detach().numpy(), y_2.detach().numpy(), color="y", s=4)
        # plt.show()
        lambda_alpha = 2
        lambda_beta = 1e-4
        loss = (p2d - g).abs().sum() / (68*bs) \
               + lambda_alpha * self.alpha.pow(2).sum()\
               + lambda_beta * self.delta.pow(2).sum() / bs
        return loss

    def init_Rt(self):
        R = torch.eye(3)
        # rotate 180 degrees!
        # R[1, 1] = -1
        t = torch.zeros(3).view(-1, 1)
        t[2] = -400
        self.R = nn.Parameter(R)
        self.t = nn.Parameter(t)
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
        if y < 0 or y >= im_height - 1 or x < 0 or x >= im_width - 1:
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


def get_ground_truth_landmarks(img, predictor=None, visualize=False):
    if predictor is None:
        predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    ground_truth = np.array([(point.x, point.y) for point in predictor(img, dets[0]).parts()])

    if visualize:
        print("HERE")
        visualize_landmarks(img, dets, predictor)

    return ground_truth


def visualize_landmarks(img, dets, predictor):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    dlib.hit_enter_to_continue()


def exercise_4_and_5(model, optimizer, img, S_land, S_whole, face_model, triangles, number_whole_points):
    basis_shape, basis_expr = face_model

    ground_truth = get_ground_truth_landmarks(img, visualize=False)
    # norm_ground_truth, min_max_gt = normalize_points(ground_truth)

    for i in range(300):
        optimizer.zero_grad()
        loss = model(S_land, ground_truth)
        loss.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, loss.item()))

    p = torch.from_numpy(S_land).float()
    whole_p = torch.from_numpy(S_whole).float()

    # p vector where the batch is expected to be in the first
    p3d_w = model.basis_shape @ (model.alpha * model.sigma_shape) \
            + model.basis_expr @ (model.delta.flatten() * model.sigma_expr)
    p3d_w = p + torch.cat((p3d_w, torch.zeros(68, 1)), dim=1)

    # Projection
    p2d = model.project_points(p3d_w)
    # p2d = denormalize_points(p2d.detach().numpy(), min_max_gt)
    p2d = p2d.detach().numpy()

    x_1, y_1 = p2d[:, 0], p2d[:, 1]
    x_2, y_2 = ground_truth[:, 0], ground_truth[:, 1]

    plt.imshow(img)
    plt.scatter(x_1, y_1, color="b", s=4)
    plt.scatter(x_2, y_2, color="y", s=4)
    plt.show()

    # All alpha and gamma applied on all 3d points_land
    whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                     + torch.from_numpy(basis_expr) @ (model.delta.flatten() * model.sigma_expr)
    whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

    # projection all points_land
    whole_p2d = model.project_points(whole_p3d)
    whole_p2d = whole_p2d.detach().numpy()
    # whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
    # Exercise 5
    new_texture = find_corresponding_texture(whole_p2d, img)

    mesh = trimesh.base.Trimesh(
        vertices=whole_p3d.detach().numpy()[:, :3],
        faces=triangles,
        vertex_colors=new_texture
    )

    mesh_to_png("rendered_face.png", mesh)
    mesh.show()


def exercise_6(model, images, S_land, S_whole, face_model, triangles, number_whole_points):
    basis_shape, basis_expr = face_model

    ground_truths = list(map(get_ground_truth_landmarks, images))

    for m in range(1, 5, 3):
        model.alpha = nn.Parameter(torch.zeros_like(model.alpha))
        model.delta = nn.Parameter(torch.zeros(m, model.delta.shape[-1]))
        model.init_Rt()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        for i in range(m * 300):
            optimizer.zero_grad()
            loss = model(S_land, np.array(ground_truths[:m]))
            loss.backward()
            optimizer.step()
            print("Iter: {}, loss: {}".format(i, loss.item()))

        whole_p = torch.from_numpy(S_whole).float()

        # All alpha and gamma applied on all 3d points_land
        whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                         + torch.from_numpy(basis_expr) @ (model.delta[0, :].flatten() * model.sigma_expr)
        whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

        # projection all points_land
        whole_p2d = model.project_points(whole_p3d)
        whole_p2d = whole_p2d.detach().numpy()
        # whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
        # Exercise 5
        new_texture = find_corresponding_texture(whole_p2d, images[0])

        mesh = trimesh.base.Trimesh(
            vertices=whole_p3d.detach().numpy()[:, :3],
            faces=triangles,
            vertex_colors=new_texture
        )

        mesh_to_png(f"face6_{m}.png", mesh)
        #mesh.show()


def exercise7(model, filep, S_land, S_whole, face_model, triangles, number_whole_points, bs=1):
    model.delta = nn.Parameter(torch.zeros((bs, model.delta.shape[-1])))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    basis_shape, basis_expr = face_model

    video_cap = cv.VideoCapture(filep)
    frames = [video_cap.read()[1] for _ in range(bs)]

    ground_truths = np.array(list(map(get_ground_truth_landmarks, frames)))
    # norm_ground_truth, min_max_gt = normalize_points(ground_truth)

    for i in range(200):
        optimizer.zero_grad()
        loss = model(S_land, ground_truths)
        loss.backward()
        optimizer.step()
        print("Iter: {}, loss: {}".format(i, loss.item()))

    model.alpha.requires_grad = False
    whole_p = torch.from_numpy(S_whole).float()

    video_cap.release()
    video_cap = cv.VideoCapture(filep)

    for i in np.arange(100):
        print("Current frame: ", i)
        _ret, frame = video_cap.read()
        if _ret:
            ground_truth = get_ground_truth_landmarks(frame)
            # norm_ground_truth, min_max_gt = normalize_points(ground_truth)

            model.init_Rt()
            model.delta = nn.Parameter(torch.zeros(1, 20))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for j in range(300):
                optimizer.zero_grad()
                out = model(S_land, ground_truth)
                out.backward()
                optimizer.step()
                print("Iter: {}, loss: {}".format(j, out.item()))

            rigid = model.R, model.t

            # All alpha and gamma applied on all 3d points_land
            whole_p3d_base = torch.from_numpy(basis_shape) @ (model.alpha * model.sigma_shape) \
                             + torch.from_numpy(basis_expr) @ (model.delta.flatten() * model.sigma_expr)
            whole_p3d = whole_p + torch.cat((whole_p3d_base, torch.zeros(number_whole_points, 1)), dim=1)

            # projection all points_land
            whole_p2d = model.project_points(whole_p3d, rigid=rigid)
            # whole_p2d = denormalize_points(whole_p2d.detach().numpy()[:, :2], min_max_gt)
            new_texture = find_corresponding_texture(whole_p2d.detach().numpy()[:, :2], frame)

            # conv_p3d = denormalize_points(whole_p3d.detach().numpy()[:, :3], min_max_w_points)

            mesh = trimesh.base.Trimesh(
                vertices=whole_p3d.detach().numpy()[:, :3],
                faces=triangles,
                vertex_colors=new_texture
            )

            mesh_to_png("video_results/my_mesh_{}.png".format(i), mesh)

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

    do_4, do_6, do_7 = False, True, False

    # Images for exercise 4
    if do_4:
        img = dlib.load_rgb_image("faces/dan.jpg")
        # img = dlib.load_rgb_image("faces/surprise.png")
        # img = dlib.load_rgb_image("faces/exercise_6/dave1.jpg")

    # For question 6
    if do_6:
        imgs = [dlib.load_rgb_image(f"faces/exercise_7/frame{i}.jpg")[:, :450, :] for i in range(1, 5)]
        img = imgs[0]

    # Load video for exercise 7
    if do_7:
        video_filep = "faces/exercise_7/smile.mp4"
        video_cap = cv.VideoCapture(video_filep)
        _ret, img = video_cap.read()
        # TODO: Make this adaptive by getting the ground truth face landmarks!
        img = img[:, :450, :]
        video_cap.release()

    # Get shape of current input image
    im_shape = img.shape

    # Instantiate the model
    model = EnergyMin(
        mean_shape_land,
        sigma2_shape,
        basis_shape_land,
        mean_expr_land,
        sigma2_expr,
        basis_expr_land,
        im_shape
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Check initialized matrices
    print("P Matrix")
    print(model.P)
    print("V Matrix")
    print(model.V)

    # Landmark points
    points_land = mean_shape_land + mean_expr_land
    S_land = np.concatenate((points_land, np.ones((number_landmark, 1))), axis=1)

    # Total points of 3D face model
    whole_points = mean_shape + mean_expr
    S_whole = np.concatenate((whole_points, np.ones((number_whole_points, 1))), axis=1)

    # Normalize input for better convergence - Not used at the moment
    # points_land_norm, min_max_points_land = normalize_points(points_land)
    # S_land_norm = np.concatenate((points_land_norm, np.ones((number_landmark, 1))), axis=1)
    # whole_points_norm, min_max_w_points = normalize_points(whole_points, min_max_points_land)
    # S_whole_norm = np.concatenate((whole_points_norm, np.ones((number_whole_points, 1))), axis=1)

    face_model = basis_shape, basis_expr

    if do_4:
        exercise_4_and_5(model, optimizer, img, S_land, S_whole, face_model, triangles, number_whole_points)
    if do_6:
        exercise_6(model, imgs, S_land, S_whole, face_model, triangles, number_whole_points)
    if do_7:
        # BS is batch size for estimating α
        exercise7(model, video_filep, S_land, S_whole, face_model, triangles, number_whole_points, bs=20)


if __name__ == "__main__":
    main()
