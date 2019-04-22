import numpy as np
import open3d as o3d
import scipy.io
import Assignment_1.calc_IPC as IPC
import time
import pickle
import matplotlib.pyplot as plt


def estimate_transformations(sample_size, sample_technique):
    """
    TODO: Augment function for all the experimental conditions
    :return:
    """

    start_time = time.time()

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 4, 4))

    rotation = np.eye(3)
    translation = np.zeros(3)

    for i in np.arange(99):
        base, target = load_point_clouds(i)

        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]

        base_point_cloud_colors = np.asarray(base[0].colors)

        R, t, rms_errors = IPC.calc_ICP(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                        target_point_cloud_normal, base_point_cloud_colors,
                                        (sample_technique, sample_size))

        # Calc and save as affine transformation
        rotation = R.dot(rotation)
        translation = R.dot(translation) + t
        transform = np.hstack((rotation, translation.reshape((3, 1))))
        transform_affine = np.append(transform, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0)
        transformations = np.append(transformations, transform_affine.reshape((1, 4, 4)), axis=0)

    minutes_elapsed = int((time.time() - start_time) / 60)
    print(minutes_elapsed)

    np.save("Transformations/data_transformations_sample_" + str(sample_size) + "_" + sample_technique, transformations)


def load_point_clouds(index, load_only_base=False):
    """

    :param index: Index of the current base point cloud
    :param load_only_base: Index of the current base point cloud
    :return: base and target point cloud data
    """
    file_id_source = "00000000" + "{0:0=2d}".format(index + 1)
    file_id_target = "00000000" + "{0:0=2d}".format(index)

    print(file_id_source)
    # print(file_id_target)

    # Read source
    base_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_source + ".pcd")
    base_point_cloud_coords = np.asarray(base_point_cloud.points)
    base_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_source + "_normal.pcd", delimiter=' ',
                                            skip_header=11)

    if load_only_base:
        return base_point_cloud, base_point_cloud_coords, base_point_cloud_normal

    # Read target
    target_point_cloud = o3d.read_point_cloud("Data/data/" + file_id_target + ".pcd")
    target_point_cloud_coords = np.asarray(target_point_cloud.points)
    target_point_cloud_normal = np.genfromtxt("Data/data/" + file_id_target + "_normal.pcd", delimiter=' ',
                                              skip_header=11)

    base = (base_point_cloud, base_point_cloud_coords, base_point_cloud_normal)
    target = (target_point_cloud, target_point_cloud_coords, target_point_cloud_normal)

    return base, target


def reconstruct_3d(sample_size, sample_technique):
    """

    :return:
    """

    transformations = np.load(
        "Transformations/data_transformations_sample_" + str(sample_size) + "_" + sample_technique + ".npy")
    # transformations = np.load("Transformations/data_transformations_sample_5000_uniform.npy")
    print(transformations.shape)

    reconstructed_data = np.zeros((0, 3))

    for i in np.arange(99):
        base = load_point_clouds(i, True)
        base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
        A1, A1_normal = IPC.cleanInput(base_point_cloud_coords, base_point_cloud_normal)

        if i > 0:
            trans = transformations[i - 1]

            A1 = np.hstack((A1, np.ones((A1.shape[0], 1))))
            A1 = np.dot(A1, trans.T)

        reconstructed_data = np.append(reconstructed_data, A1[:, 0:3], axis=0)

    visualize_reconstructed(reconstructed_data)


# Test function for debugging. Alters transformations that they map on 99 and not on 0
# Can most likely be deleted
def rearange_transformation_order(transformations):
    n = transformations.shape[0]

    transformations_new = np.zeros((0, 3, 4))

    for i in np.arange(100)[::-1]:
        rotation = np.eye(3)
        translation = np.zeros(3)
        for j in np.arange(i, n):
            trans = transformations[j]
            R = trans[:, 0:3]
            t = trans[:, 3]

            rotation = R.dot(rotation)
            translation = R.dot(translation) + t

        trans_new = np.hstack((rotation, translation.reshape((3, 1))))
        transformations_new = np.append(transformations_new, trans_new.reshape((1, 3, 4)), axis=0)

    np.save("Transformations/transformations_rewind", np.flip(transformations_new, 0))


def visualize_reconstructed(reconstructed):
    """

    :param reconstructed:
    :return:
    """
    point_cloud_rec = o3d.PointCloud()
    point_cloud_rec.points = o3d.Vector3dVector(reconstructed)
    o3d.draw_geometries([point_cloud_rec])


def run_experiments_ex_2():
    """
    Test ICP algorithm w.r.t. (a) accuracy, (b) speed, (c) stability and (d) tolerance to noise by changing the
    point selection technique: (a) all the points, (b) uniform sub-sampling, (c) random sub-sampling in each iteration
    and (d) sub-sampling more from informative regions
    :return:
    """

    # Get point clouds
    base, target = load_point_clouds(0)
    base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
    target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]
    base_point_cloud_colors = np.asarray(base[0].colors)

    # Set experiment parameters
    sampling_techniques = ["all", "uniform", "rnd_i", "inf_reg"]
    # sampling_techniques = ["uniform", "rnd_i", "inf_reg"]
    # sampling_techniques = ["inf_reg"]
    sample_size = 5000

    # Save Results
    results = [[], [], [], []]

    for i, samp_tech in enumerate(sampling_techniques):
        print("Current sampling", i)

        # Save both total time as well as for each iteration
        minutes_elapsed_total = [0, []]
        rms_errors_iter = []
        # We go over 5 runs to get a better estimate over the time and convergence
        for j in range(3):
            print("Current iter", j)
            start_time = time.time()
            R, t, rms_errors = IPC.calc_ICP(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                            target_point_cloud_normal, base_point_cloud_colors,
                                            (samp_tech, sample_size))

            # TODO: Implement stability and tolerance to noise

            minutes_elapsed_iter = (time.time() - start_time) / 60
            minutes_elapsed_total[0] += minutes_elapsed_iter
            minutes_elapsed_total[1].append(minutes_elapsed_iter)
            print("Time this iteration", minutes_elapsed_iter)

            rms_errors_iter.append(rms_errors)

        # Save all the necessary Results
        results[i] = (rms_errors_iter, minutes_elapsed_total)

    with open('Results/results_2_1.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_resutls_ex_2():
    with open('Results/results_2_1.pkl', 'rb') as f:
        results = pickle.load(f)

    # sampling_techniques = ["all", "uniform", "rnd_i", "inf_reg"]
    sampling_techniques = ["inf_reg"]

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

    # plt.legend(handles=[saved_plots[0][0], saved_plots[1][0], saved_plots[2][0], saved_plots[3][0]],
    #            labels=['All Data', 'Uniform', 'Random each Iteration', 'Informative Regions'], prop={'size': 15})

    plt.legend(handles=[saved_plots[0][0]], labels=['Informative Regions'], prop={'size': 15})
    plt.show()


def run_experiments_ex_3():
    """
    - 3.1 (b) camera pose and merge the Results using every 2 nd , 4 th , and 10 th frames
    - 3.2 ...
    :return:
    """
    # TODO: Run experiments for exercise 3.X

    pass


# base_point_cloud = scipy.io.loadmat('Data/source.mat')["source"].T
# target_point_cloud = scipy.io.loadmat('Data/target.mat')["target"].T

# R, t = IPC.calc_IPC(base_point_cloud, target_point_cloud)


# estimate_transformations(5000, "inf_reg")
# reconstruct_3d(5000, "inf_reg")

# base = load_point_clouds(20, True)

run_experiments_ex_2()
# plot_resutls_ex_2()
