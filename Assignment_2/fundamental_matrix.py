import numpy as np
import cv2 as cv
import glob
from scipy.linalg import null_space
from matplotlib import pyplot as plt


# Since we use Python, instead of using VLFeat we perform feature matching with ORB and BFMatcher. The corresponding
# code is based on
# (https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html)
# (https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)


def ransac(N, keypoints_1, keypoints_2, t, estimator="norm_eight_point", verbose=True):
    """
    Performs the RANSAC in order to estimate the fundamental matrix F.
    (Or if estimator="baseline_homography", we calculate the homography between the two images).
    :param N: Number of iterations
    :param keypoints_1: Corresponding keypoints in the first image
    :param keypoints_2: Corresponding keypoints in the second image
    :param t: Threshold for nearest neighbor estimation
    :param estimator:
    :param verbose:

    :return: Best transformation
    """
    best_model = None
    best_fit = 1

    # Get points for testing F
    # points, points_ = get_matching_points(matches, make_homogeneous(keypoints_1), make_homogeneous(keypoints_2))

    # Need 8 unknowns to solve for fundamental matrix. More will only introduce additional outliers
    if estimator == "baseline_homography":
        sample_size = 4
    elif estimator == "eight_point" or estimator == "norm_eight_point":
        sample_size = 8
    else:
        print("No estimation technique specified")
        return None

    for n in np.arange(N):
        if verbose:
            print("Current Iteration", n)
        sample = np.random.choice(keypoints_1.shape[0], sample_size, replace=False)
        sample_kp_im1 = keypoints_1[sample]
        sample_kp_im2 = keypoints_2[sample]
        if estimator == "baseline_homography":
            F = fit_model_to_sample(sample_kp_im1, sample_kp_im2)
        elif estimator == "eight_point":
            F = eight_point_algorithm(sample_kp_im1, sample_kp_im2)
        elif estimator == "norm_eight_point":
            F = normalized_eight_point_algorithm(sample_kp_im1, sample_kp_im2)
        else:
            print("Wrong estimator?")
            return None

        inliers = estimate_inliers(F, keypoints_1, keypoints_2, t)
        number_inliers = np.where(inliers)[0].shape[0]
        if verbose:
            print("Number Inliers", number_inliers)

        if number_inliers > best_fit:
            best_fit = number_inliers

            if estimator == "baseline_homography":
                F = fit_model_to_sample(keypoints_1[inliers, :2], keypoints_2[inliers, :2])
            elif estimator == "eight_point":
                F = eight_point_algorithm(keypoints_1[inliers, :], keypoints_2[inliers, :])
            elif estimator == "norm_eight_point":
                F = normalized_eight_point_algorithm(keypoints_1[inliers, :], keypoints_2[inliers, :])
            else:
                print("Wrong estimator?")
                return None

            best_model = F

            # if number_inliers >= len(matches):
            #     return best_model, keypoints_1, keypoints_2

    if verbose:
        print("Best number of inliers:", best_fit)
    return best_model, keypoints_1, keypoints_2


def estimate_inliers(F, points, points_, t):
    """
    Determines inliers for the RANSAC algorithm to estimate the fundamental matrix F. The variable names are close to
    the procedure given in the assignment sheet.
    :param F:
    :param points:
    :param points_:
    :param t:
    :return:
    """
    numerator = np.asarray([np.square(points_[i].T @ F @ points[i]) for i in np.arange(points.shape[0])])  # numerator

    # F*p_i (1)
    Fp1_2 = np.asarray([np.square((F @ points[i])[0]) for i in np.arange(points.shape[0])])
    # F*p_i (2)
    Fp2_2 = np.asarray([np.square((F @ points[i])[1]) for i in np.arange(points.shape[0])])
    # F.T*p_i' (1)
    Fp1__2 = np.asarray([np.square((F @ points_[i])[0]) for i in np.arange(points_.shape[0])])
    # F.T*p_i' (2)
    Fp2__2 = np.asarray([np.square((F @ points_[i])[1]) for i in np.arange(points_.shape[0])])

    denominator = Fp1_2 + Fp2_2 + Fp1__2 + Fp2__2

    d_is = numerator / denominator

    return d_is < t


def estimate_inliers_for_homography(F, points, points_, t):
    """
    In case we want to determine the homography between two images with RANSAC, we use this function to determine inliers
    :param F:
    :param points:
    :param points_:
    :param t:
    :return:
    """
    predicted_points = (F @ points.T).T

    inliers = []
    for i, p_point in enumerate(predicted_points):
        gt_point = points_[i]

        diff = np.absolute(p_point - gt_point)

        if diff[0] < t and diff[1] < t:
            inliers.append(i)

    return np.asarray(inliers)


def find_epipolar_lines(F, keypoints_1, keypoints_2):
    """
    Finds epipolar lines by using matching keypoints and the fundamental matrix F.
    :param F:
    :param keypoints_1:
    :param keypoints_2:
    :return:
    """
    l_rs = [F @ keypoints_1[i] for i in np.arange(keypoints_1.shape[0])]
    l_ls = [F.T @ keypoints_2[i] for i in np.arange(keypoints_2.shape[0])]

    return np.asarray(l_ls), np.asarray(l_rs)


def find_epipoles(F):
    """
    Finds the left and right nullspace of F to get the position of the epipoles.
    :param F:
    :return:
    """
    return np.asarray(null_space(F.T)), np.asarray(null_space(F))


def estimate_fundamental_matrix(matches, keypoints_1, keypoints_2, normalized=True):
    """
    Estimates fundamental matrix given two sets of keypoints and their matches without using RANSAC.
    :param matches:
    :param keypoints_1:
    :param keypoints_2:
    :return:
    """
    sample = np.random.choice(len(matches), 8, replace=False)
    points = keypoints_1[sample]
    points_ = keypoints_2[sample]
    if normalized:
        return normalized_eight_point_algorithm(points, points_), keypoints_1, keypoints_2
    else:
        return eight_point_algorithm(points, points_), keypoints_1, keypoints_2


def normalized_eight_point_algorithm(keypoints_1, keypoints_2):
    """
    Performs the normalized eight point algorithm for calculating the fundamental matrix F of two arrays of keypoints.
    The variable names in the code are close to the variables in the formulas in the assignment sheet.
    :param keypoints_1:
    :param keypoints_2:
    :return:
    """
    X, Y = keypoints_1[:, 0], keypoints_1[:, 1]
    X_, Y_ = keypoints_2[:, 0], keypoints_2[:, 1]  # X_: X', Y_: Y'

    mx, my = np.mean(X), np.mean(Y)
    mx_, my_ = np.mean(X_), np.mean(Y_)

    d = np.mean(np.sqrt(np.square(X - mx) + np.square(Y - my)))
    d_ = np.mean(np.sqrt(np.square(X_ - mx_) + np.square(Y_ - my_)))

    sqrt2_d = np.sqrt(2) / d
    sqrt2_d_ = np.sqrt(2) / d_

    mT = np.asarray([[sqrt2_d, 0, -mx * sqrt2_d], [0, sqrt2_d, -my * sqrt2_d], [0, 0, 1]])
    mT_ = np.asarray([[sqrt2_d_, 0, -mx_ * sqrt2_d_], [0, sqrt2_d_, -my_ * sqrt2_d_], [0, 0, 1]])

    keypoints_1_norm = (mT @ keypoints_1.T).T
    keypoints_2_norm = (mT_ @ keypoints_2.T).T

    F_norm = eight_point_algorithm(keypoints_1_norm, keypoints_2_norm)

    return mT_.T @ F_norm @ mT


def eight_point_algorithm(keypoints_1, keypoints_2):
    """
    Performs the eight point algorithm for calculating the fundamental matrix F of two arrays of keypoints. The different
    steps are related to those given in the assignment sheet.
    :param keypoints_1:
    :param keypoints_2:
    :return:s
    """
    # Fit model to sample
    A = np.zeros((0, 9))
    # Create matrix A
    for i in np.arange(keypoints_1.shape[0]):
        x, y, _ = keypoints_1[i]
        x_, y_, _ = keypoints_2[i]  # x_: x', y_: y'

        row = np.asarray([x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1])
        A = np.append(A, row.reshape((1, 9)), axis=0)

    U, D, Vt = np.linalg.svd(A, full_matrices=False)

    _F = Vt[-1, :].T.reshape(3, 3).T
    Uf, Df, Vft = np.linalg.svd(_F)
    Df[-1] = 0
    return Uf @ np.diag(Df) @ Vft


def find_matches(image_1, image_2, return_descriptors=False):
    """
    Finds matches between two images. For this purpose we use ORB keypoints and the BFmatcher
    https://docs.opencv.org/3.4.4/db/d95/classcv_1_1ORB.html
    https://docs.opencv.org/3.4.4/d3/da1/classcv_1_1BFMatcher.html
    :param image_1:
    :param image_2:
    :param return_descriptors:
    :return:
    """
    # Initiate ORB detector and BFMatcher
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

    # find the keypoints with ORB
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

    # Match descriptors.
    # queryIdx refers to keypoints_1, trainIdx refers to keypoints_2
    _matches = bf.match(descriptors_1, descriptors_2)

    if not return_descriptors:
        # Sort them in the order of their distance
        return sorted(_matches, key=lambda x: x.distance), keypoints_1, keypoints_2
    else:
        return sorted(_matches, key=lambda x: x.distance), keypoints_1, keypoints_2, descriptors_1, descriptors_2


def get_matching_points(matches, keypoints_1, keypoints_2):
    """
    Aligns (sorts) two arrays of keypoints depending on where they match.
    :param matches:
    :param keypoints_1:
    :param keypoints_2:
    :return:
    """
    image1_indices = [match.queryIdx for match in matches]
    image2_indices = [match.trainIdx for match in matches]

    return keypoints_1[image1_indices], keypoints_2[image2_indices]


def make_homogeneous(array):
    """
    Transforms a non-homogeneous point matrix into a homogeneous one.
    :param array:
    :return:
    """
    return np.int32(np.hstack((array, np.ones((array.shape[0], 1)))))


def homogeneous_to_2D_point(array):
    """
    Transforms a homogeneous point into its 2D non-homogeneous representation.
    :param array:
    :return:
    """
    ac = array[0] / array[2]
    bc = array[1] / array[2]
    return np.asarray([ac, bc]).reshape((2, 1))


def homogeneous_to_2D(array):
    """
    Transforms a matrix of homogeneous points into their 2D non-homogeneous representation.
    :param array:
    :return:
    """
    length = array.shape[0]
    ac = array[:, 0] / array[:, 2]
    bc = array[:, 1] / array[:, 2]
    return np.hstack((ac.reshape(length, 1), bc.reshape(length, 1)))


def check_fundamental_matrix(F, keypoints_1, keypoints_2, threshold):
    """
    Tests if the fundamental matrix F is correct (x'.T @ F @ x = 0) For each point where this is approx. the case we
    return the indices.
    :param F:
    :param keypoints_1:
    :param keypoints_2:
    :param threshold:
    :return:
    """
    zeros_hopefully = np.asarray([keypoints_2[i].T @ F @ keypoints_1[i] for i in np.arange(keypoints_1.shape[0])])
    indices = zeros_hopefully < threshold
    return keypoints_1[indices], keypoints_2[indices]


def get_coordinates_from_line(epls, image_shape):
    """
    Gets coefficients a, b, c of a the equation of a line => a*x + b*y + c = 0 and calculates the points of the line
    in the given image.
    :param epls:
    :param image_shape:
    :return:
    """
    length = epls.shape[0]
    Y = image_shape[0] + 1

    # Point (x, -1)
    pt1_x = np.asarray([(epls[i][1] - epls[i][2]) / epls[i][0] for i in np.arange(epls.shape[0])])
    pt1_y = np.ones((length, 1)) * -1
    pt1 = np.hstack((pt1_x.reshape(length, 1).astype(np.int64), pt1_y.reshape(length, 1).astype(np.int64)))

    # Point (x, Y+1)
    pt2_x = np.asarray([(epls[i][1] * -Y - epls[i][2]) / epls[i][0] for i in np.arange(epls.shape[0])])
    pt2_y = np.ones((length, 1)) * Y
    pt2 = np.hstack((pt2_x.reshape(length, 1).astype(np.int64), pt2_y.reshape(length, 1).astype(np.int64)))

    return pt1, pt2


def estimate_homography(image_1, image_2, matches, keypoints_1, keypoints_2, N, t, good_enough):
    """
    Estimates the homography between two images.
    :param image_1:
    :param image_2:
    :param matches:
    :param keypoints_1:
    :param keypoints_2:
    :param N:
    :param t:
    :param good_enough:
    :return:
    """
    best_model, points, points_ = ransac(N, matches, keypoints_1, keypoints_2, t, estimator="baseline_homography")

    # im_matches = cv.drawMatches(image_1, kp_des_1[0], image_2, kp_des_2[0], matches[0:8], None,
    #                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matched_keypoints_1, matched_keypoints_2 = get_matching_points(matches, keypoints_1, keypoints_2)
    transformed_keypoints = (best_model @ make_homogeneous(matched_keypoints_1).T).T

    # for k in np.arange(keypoints_1_np.shape[0]):
    for k in np.arange(10):
        cv.circle(image_1, (int(matched_keypoints_1[k][0]), int(matched_keypoints_1[k][1])), 4, (0, 255, 0), -1)
        # cv.circle(image_1, (int(transformed_keypoints[k][0]), int(transformed_keypoints[k][1])), 2, (0, 0, 255), -1)
        cv.circle(image_2, (int(matched_keypoints_2[k][0]), int(matched_keypoints_2[k][1])), 4, (0, 255, 0), -1)
        cv.circle(image_2, (int(transformed_keypoints[k][0]), int(transformed_keypoints[k][1])), 2, (0, 0, 255), -1)

    while cv.waitKey(30):
        cv.imshow("kp 1", image_1)
        cv.imshow("kp 2", image_2)
        # cv.imshow("matches", img3)


def fit_model_to_sample(keypoints_1, keypoints_2):
    """
    Calculates the Homography H. Supposedly not important for the assignment. Implemented because of exercise 3 point
    4 "Perform RANSAC to estimate the homography between images."
    :param keypoints_1:
    :param keypoints_2:
    :return:
    """
    print(keypoints_1.shape, keypoints_1.shape)
    # Fit model to sample
    A = np.zeros((0, 8))
    b = np.zeros((0, 1))  # zero vector of sample size
    # Create matrix A

    for i in np.arange(keypoints_1.shape[0]):
        x, y = keypoints_1[i]
        x_, y_ = keypoints_2[i]  # x_: x', y_: y'

        row_1 = np.asarray([x, y, 1, 0, 0, 0, -x * x_, -y * x_])
        row_2 = np.asarray([0, 0, 0, x, y, 1, -x * y_, -y * y_])

        b_1 = x_
        b_2 = y_

        A = np.append(A, row_1.reshape((1, 8)), axis=0)
        A = np.append(A, row_2.reshape((1, 8)), axis=0)

        b = np.append(b, b_1.reshape((1, 1)), axis=0)
        b = np.append(b, b_2.reshape((1, 1)), axis=0)

    f = np.linalg.inv(A.T @ A) @ A.T @ b
    return np.append(f, np.asarray([[1]]), axis=0).reshape((3, 3))


def experiments_exercise_3(image_data, run_experiments=False):
    """
    Performs the experiments for exercise 3. After loading two consecutive images, keypoints and descriptors are
    estimated with which the fundamental matrix F and epipolar lines are determined.
    :param image_data: All image data
    :return:
    """

    np.random.seed(0)

    image_1 = image_data[0]
    image_2 = image_data[1]
    image_size = image_1.shape
    matches, keypoints_1, keypoints_2 = find_matches(image_1, image_2)

    # Convert opencv keypoint representation in numpy array
    keypoints_1_np = np.int32(cv.KeyPoint.convert(keypoints_1))
    keypoints_2_np = np.int32(cv.KeyPoint.convert(keypoints_2))
    keypoints_1_np, keypoints_2_np = get_matching_points(matches, keypoints_1_np, keypoints_2_np)

    # Points: (x, y) and Points_: (x', y') (see Assignment)
    keypoints_1_np, keypoints_2_np = make_homogeneous(keypoints_1_np), make_homogeneous(keypoints_2_np)

    # Compute average error for experiments
    if run_experiments:
        # (0): Plain eight point alg; (1): Norm eight-point alg; for both idx 0: errors, idx 1: fundamental matrix
        # (2): Norm eight-point w. Ransac, idx 0: errors, idx 1: fundamental matrix, idx 2: number of inliers
        overall_errors = [([], []), ([], []), ([], [], [])]

        # False: Estimate fundamental matrix without normalization
        # True: Estimate fundamental matrix with normalization
        # "Ransac": Use Ransac and normalization to estimate fundamental matrix
        condition = [False, True, "Ransac"]
        save_description = ["eight", "norm_eight", "ransac"]
        plot_description = ['Eight-point algorithm', 'Normalized Eight-point algorithm',
                            'Norm. Eight-point alg. with Ransac']

        for i, cond in enumerate(condition):
            img1 = np.copy(image_1)
            img2 = np.copy(image_2)
            for j in np.arange(100):
                if cond == "Ransac":
                    F, points, points_ = ransac(100, keypoints_1_np, keypoints_2_np, 0.05, estimator="norm_eight_point")
                else:
                    F, points, points_ = estimate_fundamental_matrix(matches, keypoints_1_np, keypoints_2_np, cond)

                l_ls, l_rs = find_epipolar_lines(F, points, points_)
                average_error = calc_average_error(l_ls, l_rs, points, points_)
                overall_errors[i][0].append(average_error)
                overall_errors[i][1].append(F)

            best_idx = np.argmin(overall_errors[i][0])
            F = overall_errors[i][1][best_idx]

            l_ls, l_rs = find_epipolar_lines(F, keypoints_1_np, keypoints_2_np)
            pt1_l, pt2_l = get_coordinates_from_line(l_ls, image_size)
            pt1_r, pt2_r = get_coordinates_from_line(l_rs, image_size)

            # cope with overflow errors
            ii32 = np.iinfo(np.int32)
            min_int = ii32.min
            max_int = ii32.max

            for k in np.arange(points.shape[0]):
                ptl1_x, ptl1_y = pt1_l[k][0], pt1_l[k][1]
                ptl2_x, ptl2_y = pt2_l[k][0], pt2_l[k][1]
                ptr1_x, ptr1_y = pt1_r[k][0], pt1_r[k][1]
                ptr2_x, ptr2_y = pt2_r[k][0], pt2_r[k][1]

                if not all(min_int <= i <= max_int for i in
                           [ptl1_x, ptl1_y, ptl2_x, ptl2_y, ptr1_x, ptr1_y, ptr2_x, ptr2_y]):
                    continue

                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv.circle(img1, (int(points[k][0]), int(points[k][1])), 4, (0, 255, 0), -1)
                cv.circle(img2, (int(points_[k][0]), int(points_[k][1])), 4, (0, 255, 0), -1)

                cv.line(img1, (pt1_l[k][0], pt1_l[k][1]), (pt2_l[k][0], pt2_l[k][1]), color, thickness=1, lineType=8)
                cv.line(img2, (pt1_r[k][0], pt1_r[k][1]), (pt2_r[k][0], pt2_r[k][1]), color, thickness=1, lineType=8)

            f = plt.figure(dpi=400)
            f.suptitle("Epipolar lines estimated with the \n {}".format(plot_description[i]), fontweight='bold',
                       fontsize=16)
            ax1 = f.add_subplot(121)
            ax2 = f.add_subplot(122)
            ax1.set_title('Left image')
            ax1.imshow(img1)
            ax2.set_title('Right image')
            ax2.imshow(img2)
            plt.savefig("Plots/Epipolar_lines_plot_{}.png".format(save_description[i]), dpi=300)
            plt.show()

        sorted_error_eight = sorted(overall_errors[0][0], reverse=True)
        sorted_error_normeight = sorted(overall_errors[1][0], reverse=True)
        sorted_error_ransac = sorted(overall_errors[2][0], reverse=True)

        f, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 6))
        f.suptitle("Average Error over the different approaches", fontweight='bold', fontsize=18)

        ep = axs[0].plot(sorted_error_eight, 'm')
        nep = axs[1].plot(sorted_error_normeight, 'c')
        rans = axs[2].plot(sorted_error_ransac, 'g')

        plt.legend(handles=[ep[0], nep[0], rans[0]],
                   labels=['Eight-point algorithm', 'Normalized Eight-point algorithm',
                           'Norm. Eight-point alg. with Ransac'], prop={'size': 12})

        plt.savefig("Plots/Average_error_results.png", dpi=300)
        plt.show()

        print("----------------------------------------------------------")
        print("-------------------------RESULTS--------------------------")

        print("Eight point:")
        print("min average error", sorted_error_eight[-1])
        print("avg average error", np.mean(sorted_error_eight))
        print("max average error", sorted_error_eight[0])
        print("Norm. Eight-point:")
        print("min average error", sorted_error_normeight[-1])
        print("avg average error", np.mean(sorted_error_normeight))
        print("max average error", sorted_error_normeight[0])
        print("RANSAC:", sorted_error_ransac[-1])
        print("min average error", sorted_error_ransac[-1])
        print("avg average error", np.mean(sorted_error_ransac))
        print("max average error", sorted_error_ransac[0])
        print("----------------------------------------------------------")
        print("----------------------------------------------------------")

    # Normal estimation for testing with visualization afterwards
    else:
        F, points, points_ = ransac(100, points, points_, 0.05, estimator="norm_eight_point")
        # F, points, points_ = estimate_fundamental_matrix(matches, points, points_, False)
        # F, points, points_ = estimate_fundamental_matrix(matches, points, points_, True)

        print("Check Fundamental Matrix")
        points, points_ = check_fundamental_matrix(F, points, points_, 0.000001)

        l_ls, l_rs = find_epipolar_lines(F, points, points_)
        le, re = find_epipoles(F)

        print("Epipolar lines check")
        print(l_ls[0].T @ points[0])
        print(l_rs[0].T @ points_[0])

        print("Epipole check")
        print(points_[0] @ F @ le)
        print(re.T @ F @ points[0])

        print("Epipole")
        le2d = le[0] / le[2], le[1] / le[2]
        re2d = re[0] / re[2], re[1] / re[2]
        print(le2d)
        print(re2d)

        pt1_l, pt2_l = get_coordinates_from_line(l_ls, image_size)
        pt1_r, pt2_r = get_coordinates_from_line(l_rs, image_size)

        # Code for visualizing the matches
        # im_matches = cv.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:100], None,
        #                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        for k in np.arange(points.shape[0]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv.circle(image_1, (int(points[k][0]), int(points[k][1])), 4, (0, 255, 0), -1)
            cv.circle(image_2, (int(points_[k][0]), int(points_[k][1])), 4, (0, 255, 0), -1)

            cv.line(image_1, (pt1_l[k][0], pt1_l[k][1]), (pt2_l[k][0], pt2_l[k][1]), color, thickness=1, lineType=8)
            cv.line(image_2, (pt1_r[k][0], pt1_r[k][1]), (pt2_r[k][0], pt2_r[k][1]), color, thickness=1, lineType=8)

            # cv.line(image_1, (le2d[0], le2d[1]), (points[k][0], points[k][1]), color, thickness=1, lineType=8)
            # cv.line(image_2, (re2d[0], re2d[1]), (points_[k][0], points_[k][1]), color,
            #         thickness=1, lineType=8)

        # cv.circle(image_1, (int(le2d[0]), int(le2d[0])), 4, (255, 255, 0), -1)
        # cv.circle(image_2, (int(re2d[0]), int(re2d[0])), 4, (255, 255, 0), -1)

        while cv.waitKey(30):
            cv.imshow("kp 1", image_1)
            cv.imshow("kp 2", image_2)
            # cv.imshow("matches", im_matches)


def calc_average_error(l_ls, l_rs, points, points_):
    """
    Calculates the average error (point – epipolar line distance) for run combining the errors for the left and right image
    :param l_ls: Epipolar lines left image
    :param l_rs: Epipolar lines right image
    :param points: Points left image
    :param points_: Points right image
    :return: Average Error
    """
    errors = []
    for i in np.arange(l_ls.shape[0]):
        left_line, right_line = l_ls[i], l_rs[i]
        left_point, right_point = points[i], points_[i]

        left_error = shortest_distance(left_point, left_line[0], left_line[1], left_line[2])
        right_error = shortest_distance(right_point, right_line[0], right_line[1], right_line[2])

        errors.append(left_error)
        errors.append(right_error)

    return np.mean(np.asarray(errors))


def shortest_distance(point, a, b, c):
    """
    Source: https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
    :param x1:
    :param y1:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return abs((a * point[0] + b * point[1] + c)) / (np.sqrt(a * a + b * b))


if __name__ == "__main__":
    image_data = [cv.imread(image) for image in sorted(glob.glob("Data/House/*.png"))]
    experiments_exercise_3(image_data, run_experiments=True)
