import numpy as np
import matplotlib.pyplot as plt
from icp import calc_icp, estimate_transformations, reconstruct_3d, load_point_cloud, visualize_points, clean_input


def run_experiments_ex_3_1(visualize=False):
    """
    - 3.1 (a) estimate camera pose with all the points
    - 3.1 (b) estimate camera pose and merge the Results using every 2 nd , 4 th , and 10 th frames
    :return:
    """
    for stride in [1, 2, 4, 10]:
        estimate_transformations(5000, "uniform", stride)

    if visualize:
        for stride in [1, 2, 4, 10]:
            reconstruct_3d(5000, "uniform", stride)


def run_experiments_ex_3_2(sample_size, sample_technique):
    # Point cloud growing over consecutive frames
    accumulated_target_coords = np.zeros((0, 3))
    accumulated_target_normals = np.zeros((0, 4))

    # Keeps track of the transformations across consecutive frames. e.g entry 0: frame 0 to 1
    transformations = np.zeros((0, 4, 4))

    plot_errors = []

    for i in np.arange(0, 99):
        if i == 0:
            # Get first base and target pair
            base = load_point_cloud(i + 1)
            target = load_point_cloud(i)
            base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
            target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]
            base_point_cloud_colors = np.asarray(base[0].colors)

            _rot, _trans, errors = calc_icp(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                            target_point_cloud_normal, base_point_cloud_colors,
                                            sample_technique, sample_size)

        else:
            base = load_point_cloud(i + 1)

            if base is None:
                break

            base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
            base_point_cloud_colors = np.asarray(base[0].colors)

            _rot, _trans, errors = calc_icp(base_point_cloud_coords, accumulated_target_coords, base_point_cloud_normal,
                                            accumulated_target_normals, base_point_cloud_colors, sample_technique,
                                            sample_size)

        plot_errors.append(errors[-1])

        transform = np.hstack((_rot, _trans.reshape((3, 1))))
        transform_rigid = np.append(transform, np.asarray([0, 0, 0, 1]).reshape((1, 4)), axis=0)
        transformations = np.append(transformations, transform_rigid.reshape((1, 4, 4)), axis=0)

        base_transformed = (base_point_cloud_coords - transform_rigid[:3, 3]) @ transform_rigid[:3, :3]

        accumulated_target_coords = np.append(accumulated_target_coords, base_transformed[:, 0:3], axis=0)
        accumulated_target_normals = np.append(accumulated_target_normals, base_point_cloud_normal, axis=0)

        if i == 0:
            accumulated_target_coords = np.append(accumulated_target_coords, target_point_cloud_coords, axis=0)
            accumulated_target_normals = np.append(accumulated_target_normals, target_point_cloud_normal, axis=0)

        current_acc_length = accumulated_target_coords.shape[0]
        if current_acc_length > 200000:
            exclude_indices = np.arange(0, int(current_acc_length * 0.25), 2)
            accumulated_target_coords = np.delete(accumulated_target_coords, exclude_indices, 0)
            accumulated_target_normals = np.delete(accumulated_target_normals, exclude_indices, 0)

        np.save(
            "Transformations/data_transformations_3_2_sample_" + str(sample_size) + "_" + sample_technique + "_fg1",
            transformations)

        np.save("Transformations/rms_error_3_2_sample_{}_{}_fg{}".format(str(sample_size), sample_technique, str(1)),
                np.asarray(plot_errors))


def reconstruct_3_2():
    transformations = np.load("Transformations/data_transformations_3_2_sample_4000_uniform_fg1.npy")

    reconstruction = np.zeros((0, 3))
    # Small hack when the frame gap leads to the transformation being shorter than the total frames
    for i, j in enumerate(range(0, 25, 1)):
        cloud = load_point_cloud(j)
        if cloud is None:
            break
        new_points, _ = clean_input(cloud[1], cloud[2])

        if j > 0:
            trans = transformations[i - 1]
            reconstruction = (reconstruction - trans[:3, 3]) @ trans[:3, :3]
        reconstruction = np.vstack((reconstruction, new_points[:, :3]))

    visualize_points(reconstruction)


def plot_final_rmses(ex_3_2=False):
    def _gplot():
        textwidth = 452. / 72
        golden_mean = (5 ** .5 - 1) / 2
        fig = plt.figure(figsize=(textwidth, textwidth * golden_mean))
        return fig.add_subplot(111)

    if not ex_3_2:
        ax = _gplot()
        ax.set_title("Final RMSE across frames", fontweight='bold', fontsize=16)
        ax.set_ylim(0.05, .45)
        ax.set_xlabel('$Frames$', fontsize=14)
        ax.set_ylabel('$RMSE$', fontsize=14)

        plots = []
        colors = ["b", "y", "r", "c"]
        for i, stride in enumerate([1, 2, 4, 10]):
            rms = np.load("Transformations/rms_error_sample_{}_{}_fg{}.npy".format(
                str(5000), "uniform", stride))
            x = stride * np.arange(rms.size)
            plot = ax.plot(x, rms, color=colors[i])
            plots.append(plot)

        plt.legend(handles=[plots[0][0], plots[1][0], plots[2][0], plots[3][0]],
                   labels=['Stride 1', 'Stride 2', 'Stride 4', 'Stride 10'], prop={'size': 12})
        plt.tight_layout()
        plt.savefig("Plots/rms_error_sample_{}_{}_fg{}.png".format(str(5000), "uniform", 1))
        plt.show()
    else:
        # for 3.2
        ax = _gplot()
        ax.set_title("Final RMSE for exercise 3.2", fontweight='bold', fontsize=16)
        ax.set_ylim(0.05, .45)
        ax.set_xlabel('$Frames$', fontsize=14)
        ax.set_ylabel('$RMSE$', fontsize=14)

        rms_32 = np.load("Transformations/rms_error_3_2_sample_4000_uniform_fg1.npy")
        ax.plot(rms_32, color="r")
        plt.tight_layout()
        plt.savefig("Plots/rms_error_3_2_sample_4000_uniform_fg1.png")
        plt.show()


# run_experiments_ex_3_1()
# run_experiments_ex_3_2(4000, "uniform")
# reconstruct_3_2()

# reconstruct_3d(5000, "uniform", 4)

plot_final_rmses(True)
