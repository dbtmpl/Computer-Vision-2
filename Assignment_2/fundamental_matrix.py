import numpy as np
import cv2 as cv
import glob
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from scipy import spatial


# Since we use Python, instead of using VLFeat we perform feature matching with ORB and BFMatcher. The corresponding
# code is based on
# (https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html)
# (https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)


def ransac(N, matches, keypoints_1, keypoints_2, t, estimator="norm_eight_point"):
    """
    Performs the RANSAC in order to estimate the fundamental matrix F.
    (Or if estimator="baseline_homography", we calculate the homography between the two images).
    :param N: Number of iterations
    :param matches: Matches found between both images
    :param keypoints_1: Corresponding keypoints in the first image
    :param keypoints_2: Corresponding keypoints in the second image
    :param t: Threshold for nearest neighbor estimation

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
        print("Current Iteration", n)
        sample = np.random.choice(len(matches), sample_size, replace=False)
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


def estimate_fundamental_matrix(matches, keypoints_1, keypoints_2):
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
    return normalized_eight_point_algorithm(points, points_), points, points_


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
    # return F_norm


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
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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
    print(zeros_hopefully.shape)
    print(zeros_hopefully)
    indices = zeros_hopefully < threshold
    print(np.where(indices)[0].shape[0])
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
    pt1 = np.hstack((pt1_x.reshape(length, 1), pt1_y.reshape(length, 1)))

    # Point (x, Y+1)
    pt2_x = np.asarray([(epls[i][1] * -Y - epls[i][2]) / epls[i][0] for i in np.arange(epls.shape[0])])
    pt2_y = np.ones((length, 1)) * Y
    pt2 = np.hstack((pt2_x.reshape(length, 1), pt2_y.reshape(length, 1)))

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


def experiments_exercise_3(image_data):
    """
    Performs the experiments for exercise 3. After loading two consecutive images, keypoints and descriptors are
    estimated with which the fundamental matrix F and epipolar lines are determined.
    :param image_data: All image data
    :return:
    """
    image_1 = image_data[0]
    image_2 = image_data[1]
    image_size = image_1.shape
    matches, keypoints_1, keypoints_2 = find_matches(image_1, image_2)

    # Convert opencv keypoint representation in numpy array
    keypoints_1_np = np.int32(cv.KeyPoint.convert(keypoints_1))
    keypoints_2_np = np.int32(cv.KeyPoint.convert(keypoints_2))
    keypoints_1_np, keypoints_2_np = get_matching_points(matches, keypoints_1_np, keypoints_2_np)

    # Points: (x, y) and Points_: (x', y') (see Assignment)
    points, points_ = make_homogeneous(keypoints_1_np), make_homogeneous(keypoints_2_np)

    F, points, points_ = ransac(100, matches, points, points_, 0.005, estimator="norm_eight_point")
    # F, points, points_ = estimate_fundamental_matrix(matches, points, points_)

    # OPENCV functions for testing purposes
    # F, _ = cv.findFundamentalMat(points, points_, cv.FM_LMEDS)
    # l_ls = cv.computeCorrespondEpilines(points_.reshape(-1, 1, 2), 2, F)
    # l_ls = l_ls.reshape(-1, 3)
    # l_rs = cv.computeCorrespondEpilines(points.reshape(-1, 1, 2), 2, F)
    # l_rs = l_rs.reshape(-1, 3)

    print("Check Fundamental Matrix")
    points, points_ = check_fundamental_matrix(F, points, points_, 0.0001)

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

        cv.line(image_1, (int(pt1_l[k][0]), int(pt1_l[k][1])), (int(pt2_l[k][0]), int(pt2_l[k][1])), color,
                thickness=1, lineType=8)
        cv.line(image_2, (int(pt1_r[k][0]), int(pt1_r[k][1])), (int(pt2_r[k][0]), int(pt2_r[k][1])), color,
                thickness=1, lineType=8)

    # cv.circle(image_1, (int(le2d[0]), int(le2d[0])), 4, (255, 255, 0), -1)
    # cv.circle(image_2, (int(re2d[0]), int(re2d[0])), 4, (255, 255, 0), -1)

    while cv.waitKey(30):
        cv.imshow("kp 1", image_1)
        cv.imshow("kp 2", image_2)
        # cv.imshow("matches", im_matches)

    # estimate_homography(image_1, image_2, matches, keypoints_1_np, keypoints_2_np, 1000, 3, 400)s


def chaining(image_data):
    """
    Builds up the point view matrix by chaining found keypoints together
    :param image_data:
    :return:
    """
    number_of_images = len(image_data)
    keypoints_upper_bound = 500 * 50
    point_view_matrix = np.zeros((number_of_images, 2, keypoints_upper_bound))
    information_previous_iteration = None
    number_saved_keypoints = 0

    # Initiate BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    for i in np.arange(0, number_of_images - 1):
        print("Iteration", i)
        # print("Current saved points", number_saved_keypoints)

        image_1 = image_data[i]
        image_2 = image_data[i + 1]

        # find the keypoints with ORB
        matches, keypoints_1, keypoints_2, descriptors_1, descriptors_2 = find_matches(image_1, image_2, True)

        print("How many matches")
        print(len(matches))

        keypoints_1_np = np.int32(cv.KeyPoint.convert(keypoints_1))
        keypoints_2_np = np.int32(cv.KeyPoint.convert(keypoints_2))

        keypoints_1_np, keypoints_2_np, descriptors_1, descriptors_2 = match_keypoints_and_descriptors(matches,
                                                                                                       keypoints_1_np,
                                                                                                       keypoints_2_np,
                                                                                                       descriptors_1,
                                                                                                       descriptors_2)

        if information_previous_iteration is not None:

            keypoints_previous, descriptors_previous, indices_previous = information_previous_iteration

            # print("Loose points?")
            # print(len(indices_previous))
            # print(np.unique(indices_previous).shape[0])

            _matches = bf.match(descriptors_1, descriptors_previous)
            _matches = sorted(_matches, key=lambda x: x.distance)
            survived_indices = [match.queryIdx for match in _matches]
            survived_pvm_indices = [match.trainIdx for match in _matches]
            current_pvm_indices = indices_previous[survived_pvm_indices]

            for j in np.arange(len(current_pvm_indices)):
                pvm_idx = current_pvm_indices[j]
                keypoint_idx = survived_indices[j]
                point_view_matrix[i][:2, pvm_idx] = keypoints_1_np[keypoint_idx].T

            # print("AFTER KP: Number of simultaneous keypoints")
            # print(np.where(point_view_matrix[i][1] > 0)[0].shape[0])

            new_keypoints = np.delete(keypoints_1_np, survived_indices, axis=0)
            number_additional_kp = new_keypoints.shape[0]
            new_indices = np.arange(number_saved_keypoints, number_saved_keypoints + number_additional_kp)
            for j, index in enumerate(new_indices):
                point_view_matrix[i][:2, index] = new_keypoints[j].T

            # print("AFTER NEW KP: Number of simultaneous keypoints")
            # print(np.where(point_view_matrix[i][1] > 0)[0].shape[0])
            # print("How many official new keypoints")
            # print(number_additional_kp)

            number_saved_keypoints += number_additional_kp

            indices_next_iteration = np.append(current_pvm_indices, new_indices, axis=0)

            # print("How many inofficial new keypoints")
            # print(len(matches) - len(survived_indices) == number_additional_kp)

            print("END: Number of simultaneous keypoints")
            print(np.where(point_view_matrix[i][1] > 0)[0].shape[0])
            # print("indices for next iteration")
            # print(indices_next_iteration.shape[0])

        else:
            point_view_matrix[i][:2, :keypoints_1_np.shape[0]] = keypoints_1_np.T
            number_saved_keypoints += len(matches)
            indices_next_iteration = np.arange(keypoints_1_np.shape[0])

        information_previous_iteration = keypoints_2_np, descriptors_2, indices_next_iteration

    point_view_matrix = point_view_matrix[:, :, :number_saved_keypoints]
    return point_view_matrix.reshape(number_of_images * 2, point_view_matrix.shape[-1])


def match_keypoints_and_descriptors(matches, keypoints_1, keypoints_2, descriptors_1, descriptors_2):
    """
    Aligns indices of keypoints and descriptors
    :param matches:
    :param keypoints_1:
    :param keypoints_2:
    :param descriptors_1:
    :param descriptors_2:
    :return:
    """
    image1_indices = [match.queryIdx for match in matches]
    image2_indices = [match.trainIdx for match in matches]
    keypoints_1, descriptors_1 = keypoints_1[image1_indices], descriptors_1[image1_indices]
    keypoints_2, descriptors_2 = keypoints_2[image2_indices], descriptors_2[image2_indices]

    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def visualize_point_view_matrix(point_view_matrix):
    """
    Transforms point view matrix into binary representation and visualizes the trajectories.
    :param point_view_matrix: Previously calculated point view matrix
    """
    binary_pvm = point_view_matrix > 0
    plt.imshow(binary_pvm, aspect=25)
    plt.savefig("Chaining_result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # image_data = [cv.imread(image) for image in sorted(glob.glob("Data/House/*.jpg"))]
    image_data = [cv.imread(image) for image in sorted(glob.glob("Data/House/*.png"))]

    # experiments_exercise_3(image_data)

    point_view_matrix = chaining(image_data)
    visualize_point_view_matrix(point_view_matrix)
