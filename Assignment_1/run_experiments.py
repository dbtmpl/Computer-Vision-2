import numpy as np
import open3d as o3d
from calc_IPC import calc_icp, clean_input
import time
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def visualize_points(points):
    """

    :param points:
    :return:
    """
    cloud = o3d.PointCloud()
    cloud.points = o3d.Vector3dVector(points)
    o3d.draw_geometries([cloud])


def load_point_cloud(index: int):
    if index >= 100:
        return None

    file_id = "00000000" + "{0:0=2d}".format(index)
    point_cloud = o3d.read_point_cloud("Data/data/" + file_id + ".pcd")
    points = np.asarray(point_cloud.points)
    normals = np.genfromtxt("Data/data/" + file_id + "_normal.pcd", delimiter=' ', skip_header=11)
    return point_cloud, points, normals


def estimate_transformations(sample_size, sample_technique, stride=1, max_frame=99):
    """
    TODO: Augment function for all the experimental conditions

    :param sample_size:
    :param sample_technique:
    :param stride: The stride between frames
    :param max_frame:
    :return:
    """

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 4, 4))

    rot = np.eye(3)
    trans = np.zeros(3)

    for i in range(0, max_frame, stride):
        base = load_point_cloud(i + stride)
        if base is None:
            break
        base_cloud, base_points, base_normals = base
        target_cloud, target_points, target_normals = load_point_cloud(i)
        base_colors = np.asarray(base_cloud.colors)

        _rot, _trans, errors = calc_icp(base_points, target_points, base_normals, target_normals, base_colors,
                                        sample_technique, sample_size)

        # Update overall transformations:
        rot = _rot @ rot
        trans = _rot @ trans + _trans

        transform = np.hstack((rot, trans.reshape((3, 1))))
        transform_affine = np.append(transform, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0)
        transformations = np.append(transformations, transform_affine.reshape((1, 4, 4)), axis=0)

    np.save("Transformations/data_transformations_sample_{}_{}_fg{}".format(
        str(sample_size), sample_technique, str(stride)), transformations)


def reconstruct_3d(sample_size, sample_technique, stride):
    """

    :param sample_size:
    :param sample_technique:
    :param stride:
    :return:
    """

    transformations = np.load("Transformations/data_transformations_sample_{}_{}_fg{}.npy".format(
        str(sample_size), sample_technique, stride))

    reconstructed_points = np.zeros((0, 3))

    # Small hack when the frame gap leads to the transformation being shorter than the total frames
    for i, j in enumerate(range(0, 99, stride)):
        base = load_point_cloud(j + stride)
        if base is None:
            break
        points, normals = clean_input(base[1], base[2])

        if j > 0:
            trans = transformations[i - 1]

            points = np.hstack((points, np.ones((points.shape[0], 1))))
            points = points @ trans.T

        reconstructed_points = np.append(reconstructed_points, points[:, 0:3], axis=0)

    visualize_points(reconstructed_points)


def run_experiments_ex_2():
    """
    Test ICP algorithm w.r.t. (a) accuracy, (b) speed, (c) stability and (d) tolerance to noise by changing the
    point selection technique: (a) all the points, (b) uniform sub-sampling, (c) random sub-sampling in each iteration
    and (d) sub-sampling more from informative regions
    :return:
    """

    # Get point clouds
    base = load_point_cloud(0)
    target = load_point_cloud(1)
    base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
    target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]
    base_point_cloud_colors = np.asarray(base[0].colors)

    # Set experiment parameters
    sampling_techniques = ["all", "uniform", "rnd_i", "inf_reg"]
    sample_size = 5000

    # Save Results
    results = [[], [], [], []]

    for i, samp_tech in enumerate(sampling_techniques):
        print("Current sampling", i)

        start_time = time.time()
        R, t, rms_errors = IPC.calc_ICP(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                        target_point_cloud_normal, base_point_cloud_colors,
                                        (samp_tech, sample_size))

        # TODO: Implement stability and tolerance to noise

        minutes_elapsed = (time.time() - start_time) / 60

        print("Minutes spent in this iteration.", minutes_elapsed)

        # Save all the necessary Results
        results[i] = (rms_errors, minutes_elapsed)

    with open('Results/results_2_1.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_resutls_ex_2():
    with open('Results/results_2_1.pkl', 'rb') as f:
        results = pickle.load(f)

    sampling_techniques = ["all", "uniform", "rnd_i", "inf_reg"]

    # plot the data
    saved_plots = []

    fig = plt.figure(figsize=(10, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Root Mean Squared Error over Iterations", fontweight='bold', fontsize=18)
    ax.set_xlabel('$Iteration$', fontsize=16)
    ax.set_ylabel('$RMS-Error$', fontsize=16)

    colors = ["r", "b", "m", "g"]

    for i, samp_tech in enumerate(sampling_techniques):
        result = results[i]
        rms_errors, minutes_elapsed = result[0], result[1]
        total_minutes_elapsed = minutes_elapsed[0]
        print("Time elapsed " + samp_tech, total_minutes_elapsed)

        # Since convergence is dynamic, hard to average over the iterations => for now get first rms
        plot_i = ax.plot(rms_errors[0], color=colors[i])
        saved_plots.append(plot_i)

    plt.legend(handles=[saved_plots[0][0], saved_plots[1][0], saved_plots[2][0], saved_plots[3][0]],
               labels=['All Data', 'Uniform', 'Random each Iteration', 'Informative Regions'], prop={'size': 15})
    plt.show()


def run_experiments_ex_3_1(visualize=False):
    """
    - 3.1 (a)
    - 3.1 (b) camera pose and merge the Results using every 2 nd , 4 th , and 10 th frames
    :return:
    """
    for frame_gap in [1, 2, 4, 10]:
        estimate_transformations(5000, "uniform", frame_gap)

    if visualize:
        for frame_gap in [1, 2, 4, 10]:
            reconstruct_3d(5000, "uniform", frame_gap)


def run_experiments_ex_3_2(sample_size, sample_technique):
    # Point cloud growing over consecutive frames
    accumulated_target_coords = np.zeros((0, 3))
    accumulated_target_normals = np.zeros((0, 4))

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 4, 4))

    rotation = np.eye(3)
    translation = np.zeros(3)

    for i in np.arange(0, 99):
        if i == 0:
            # Get first base and target pair
            base = load_point_cloud(i)
            target = load_point_cloud(i+1)
            base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
            target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]
            base_point_cloud_colors = np.asarray(base[0].colors)

            R, t, rms_errors = calc_icp(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                            target_point_cloud_normal, base_point_cloud_colors,
                                            (sample_technique, sample_size))
        else:
            base = load_point_cloud(i)

            if base is None:
                break

            base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
            base_point_cloud_colors = np.asarray(base[0].colors)

            R, t, rms_errors = calc_icp(base_point_cloud_coords, accumulated_target_coords, base_point_cloud_normal,
                                            accumulated_target_normals, base_point_cloud_colors,
                                            (sample_technique, sample_size))

        # Calc and save as affine transformation
        rotation = R.dot(rotation)
        translation = R.dot(translation) + t
        transform = np.hstack((rotation, translation.reshape((3, 1))))
        transform_affine = np.append(transform, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0)
        transformations = np.append(transformations, transform_affine.reshape((1, 4, 4)), axis=0)

        base_point_cloud_coords = np.hstack((base_point_cloud_coords, np.ones((base_point_cloud_coords.shape[0], 1))))
        base_transformed = np.dot(base_point_cloud_coords, transform_affine.T)

        accumulated_target_coords = np.append(accumulated_target_coords, base_transformed[:, 0:3], axis=0)
        accumulated_target_normals = np.append(accumulated_target_normals, base_point_cloud_normal, axis=0)

        if i == 0:
            accumulated_target_coords = np.append(accumulated_target_coords, target_point_cloud_coords, axis=0)
            accumulated_target_normals = np.append(accumulated_target_normals, target_point_cloud_normal, axis=0)

    np.save(
        "Transformations/data_transformations_sample_" + str(sample_size) + "_" + sample_technique + "_fg1",
        transformations)