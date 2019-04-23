import numpy as np
from calc_IPC import calc_icp, estimate_transformations, reconstruct_3d, load_point_cloud


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
            target = load_point_cloud(i + 1)
            base_point_cloud_coords, base_point_cloud_normal = base[1], base[2]
            target_point_cloud_coords, target_point_cloud_normal = target[1], target[2]
            base_point_cloud_colors = np.asarray(base[0].colors)

            R, t, rms_errors = calc_icp(base_point_cloud_coords, target_point_cloud_coords, base_point_cloud_normal,
                                        target_point_cloud_normal, base_point_cloud_colors,
                                        sample_technique, sample_size)
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
        "Transformations/data_transformations_3_2_sample_" + str(sample_size) + "_" + sample_technique + "_fg1",
        transformations)


# run_experiments_ex_3_1()
# run_experiments_ex_3_2(5000, "uniform")

# reconstruct_3d(5000, "uniform", 1)
# reconstruct_3d(5000, "uniform", 1)

# estimate_transformations(5000, "inf_reg")
