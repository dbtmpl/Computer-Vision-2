import numpy as np
import cv2 as cv
import glob
from scipy.linalg import null_space


# Since we use Python, instead of VLFeat we perform feature matching with ORB and BFMatcher. The corresponding code is based on
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html


def Ransac(N, matches, keypoints_1, keypoints_2, t, good_enough, model_estimator="baseline_homography"):
    """

    :param N: Number of iterations
    :param matches: Matches found between both images
    :param keypoints_1: Corresponding keypoints in the first image
    :param keypoints_2: Corresponding keypoints in the second image
    :param t: Threshold for nearest neighbor estimation

    :return: Best transformation
    """

    best_model = None
    best_fit = 0
    if model_estimator == "baseline_homography":
        sample_size = 4  # Need 8 unknowns to solve for fundamental matrix. More will only introduce more outliers
    elif model_estimator == "eight_point":
        sample_size = 8
    else:
        print("No estimation technique specified")
        return None

    for n in np.arange(N):
        print("Current Iteration", n)
        sample = np.random.choice(matches, sample_size, replace=False)
        # F = fit_model_to_sample(sample, keypoints_1, keypoints_2)
        sample_kp_im1, sample_kp_im2 = get_matching_points(sample, keypoints_1, keypoints_2)
        if model_estimator == "baseline_homography":
            F = fit_model_to_sample(sample_kp_im1, sample_kp_im2)
        elif model_estimator == "eight_point":
            F = normalized_eight_point_algorithm(sample_kp_im1, sample_kp_im2)
        else:
            print("Wrong estimator?")

        # Get points for testing F
        points, points_ = get_matching_points(matches, make_homogeneous(keypoints_1), make_homogeneous(keypoints_2))

        inliers = estimate_inliers(F, points, points_, t)
        number_inliers = np.where(inliers)[0].shape[0]
        print("Number Inliers", number_inliers)

        if number_inliers > best_fit:
            best_fit = number_inliers

            if model_estimator == "baseline_homography":
                F = fit_model_to_sample(points[inliers, :2], points_[inliers, :2])
            elif model_estimator == "eight_point":
                ptest = points[inliers, :2]
                ptest_ = points_[inliers, :2]
                F = normalized_eight_point_algorithm(points[inliers, :2], points_[inliers, :2])
            else:
                print("Wrong estimator?")
                return None

            best_model = F

            # if number_inliers > good_enough:
            #     return best_model, points, points_

    print("Best number of inliers:", best_fit)
    return best_model, points, points_


def estimate_inliers(F, points, points_, t):
    numerator = np.asarray([np.square(points_[i].T @ F @ points[i]) for i in np.arange(points.shape[0])])  # numerator

    Fp1_2 = np.asarray([np.square((F @ points[i])[0]) for i in np.arange(points.shape[0])])
    Fp2_2 = np.asarray([np.square((F @ points[i])[1]) for i in np.arange(points.shape[0])])
    Fp1__2 = np.asarray([np.square((F @ points_[i])[0]) for i in np.arange(points_.shape[0])])
    Fp2__2 = np.asarray([np.square((F @ points_[i])[1]) for i in np.arange(points_.shape[0])])

    denominator = Fp1_2 + Fp2_2 + Fp1__2 + Fp2__2

    d_is = numerator / denominator

    return d_is < t


def estimate_inliers_for_homography(F, points, points_, t):
    predicted_points = (F @ points.T).T

    inliers = []
    for i, p_point in enumerate(predicted_points):
        gt_point = points_[i]

        diff = np.absolute(p_point - gt_point)

        if diff[0] < t and diff[1] < t:
            inliers.append(i)

    return np.asarray(inliers)


def find_epipolar_lines(F, keypoints_1, keypoints_2):
    l_rs = [F @ keypoints_1[i] for i in np.arange(keypoints_1.shape[0])]
    l_ls = [F.T @ keypoints_2[i] for i in np.arange(keypoints_2.shape[0])]

    return np.asarray(l_ls), np.asarray(l_rs)


def find_epipoles(F):
    return np.asarray(null_space(F.T)), np.asarray(null_space(F))


def normalized_eight_point_algorithm(keypoints_1, keypoints_2):
    X, Y = keypoints_1[:, 0], keypoints_1[:, 1]
    X_, Y_ = keypoints_1[:, 0], keypoints_1[:, 1]  # X_: X', Y_: Y'

    mx, my = np.mean(X), np.mean(Y)
    mx_, my_ = np.mean(X_), np.mean(Y_)

    d = np.mean(np.sqrt(np.square(X - mx) + np.square(Y - my)))
    d_ = np.mean(np.sqrt(np.square(X_ - mx_) + np.square(Y_ - my_)))

    sqrt2_d = np.sqrt(2) / d
    sqrt2_d_ = np.sqrt(2) / d_

    T = np.asarray([[sqrt2_d, 0, -mx * sqrt2_d], [0, sqrt2_d, -my * sqrt2_d], [0, 0, 1]])
    T_ = np.asarray([[sqrt2_d_, 0, -mx_ * sqrt2_d_], [0, sqrt2_d_, -my_ * sqrt2_d_], [0, 0, 1]])

    keypoints_1_norm = (T @ make_homogeneous(keypoints_1).T).T
    keypoints_2_norm = (T_ @ make_homogeneous(keypoints_2).T).T

    F_norm = eight_point_algorithm(keypoints_1_norm[:, :2], keypoints_2_norm[:, :2])

    return T_.T @ F_norm @ T


def eight_point_algorithm(keypoints_1, keypoints_2):
    # Fit model to sample
    A = np.zeros((0, 9))
    # Create matrix A
    for i in np.arange(keypoints_1.shape[0]):
        x, y = keypoints_1[i]
        x_, y_ = keypoints_2[i]  # x_: x', y_: y'

        row = np.asarray([x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1])
        A = np.append(A, row.reshape((1, 9)), axis=0)

    U, D, Vt = np.linalg.svd(A, full_matrices=False)

    _F = Vt[-1, :].T.reshape(3, 3).T
    Uf, Df, Vft = np.linalg.svd(_F)
    Df[-1] = 0
    return Uf @ np.diag(Df) @ Vft


def find_matches(image_1, image_2):
    # Initiate ORB detector and BFMatcher
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # find the keypoints with ORB
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

    # Match descriptors.
    # queryIdx refers to keypoints_1, trainIdx refers to keypoints_2
    _matches = bf.match(descriptors_1, descriptors_2)

    # Sort them in the order of their distance
    return sorted(_matches, key=lambda x: x.distance), (keypoints_1, descriptors_1), (keypoints_2, descriptors_2)


def get_matching_points(matches, keypoints_1, keypoints_2):
    image1_indices = [match.queryIdx for match in matches]
    image2_indices = [match.trainIdx for match in matches]

    return keypoints_1[image1_indices], keypoints_2[image2_indices]


def make_homogeneous(array):
    return np.hstack((array, np.ones((array.shape[0], 1))))


def homogeneous_to_2D_point(array):
    ac = array[0] / array[2]
    bc = array[1] / array[2]
    return np.asarray([ac, bc]).reshape((2, 1))


def homogeneous_to_2D(array):
    length = array.shape[0]
    ac = array[:, 0] / array[:, 2]
    bc = array[:, 1] / array[:, 2]
    return np.hstack((ac.reshape(length, 1), bc.reshape(length, 1)))


def estimate_fundamental_matrix(matches, keypoints_1, keypoints_2):
    sample = np.random.choice(matches, 8, replace=False)
    points, points_ = get_matching_points(sample, keypoints_1, keypoints_2)
    return normalized_eight_point_algorithm(points, points_), make_homogeneous(points), make_homogeneous(points_)


def check_fundamental_matrix(F, keypoints_1, keypoints_2):
    zeros_hopefully = [keypoints_2[i].T @ F @ keypoints_1[i] for i in np.arange(keypoints_1.shape[0])]

    print(zeros_hopefully)


def get_coordinates_from_line(epls, image_shape):
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
    best_model, points, points_, pp_with_inliers = Ransac(N, matches, keypoints_1, keypoints_2, t,
                                                          good_enough, model_estimator="baseline_homography")

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


if __name__ == "__main__":
    image_data = [cv.imread(image) for image in sorted(glob.glob("Data/House/*.png"))]
    image_1 = image_data[0]
    image_2 = image_data[1]

    image_size = image_1.shape

    matches, kp_des_1, kp_des_2 = find_matches(image_1, image_2)

    # Convert opencv keypoint representation in numpy array
    keypoints_1_np = cv.KeyPoint.convert(kp_des_1[0])
    keypoints_2_np = cv.KeyPoint.convert(kp_des_2[0])

    print(len(matches))

    F, points, points_ = Ransac(1000, matches, keypoints_1_np, keypoints_2_np, 0.05, 500,
                                model_estimator="eight_point")

    # F, points, points_ = estimate_fundamental_matrix(matches, keypoints_1_np, keypoints_2_np)

    print("Check Fundamental Matrix")
    check_fundamental_matrix(F, points, points_)

    l_ls, l_rs = find_epipolar_lines(F, points, points_)
    le, re = find_epipoles(F)

    print("Epipolar lines check")
    print(l_ls[0].T @ points[0])
    print(l_rs[0].T @ points_[0])

    print("Epipole check")
    print(points_[0] @ F @ le)
    print(re.T @ F @ points[0])

    print("Epipole")
    print(le, re)

    pt1_l, pt2_l = get_coordinates_from_line(l_ls, image_size)
    pt1_r, pt2_r = get_coordinates_from_line(l_ls, image_size)

    for k in np.arange(points.shape[0]):
        cv.circle(image_1, (int(points[k][0]), int(points[k][1])), 4, (0, 255, 0), -1)
        cv.circle(image_2, (int(points_[k][0]), int(points_[k][1])), 4, (0, 255, 0), -1)

        cv.circle(image_1, (int(le[0]), int(le[1])), 4, (0, 255, 0), -1)
        cv.circle(image_2, (int(re[0]), int(re[1])), 4, (0, 255, 0), -1)
        #
        cv.line(image_1, (int(pt1_l[k][0]), int(pt1_l[k][1])), (int(pt2_l[k][0]), int(pt2_l[k][1])), (255, 0, 0), thickness=1,
                lineType=8)
        cv.line(image_2, (int(pt1_r[k][0]), int(pt1_r[k][1])), (int(pt2_r[k][0]), int(pt2_r[k][1])), (255, 0, 0), thickness=1,
                lineType=8)

    while cv.waitKey(30):
        cv.imshow("kp 1", image_1)
        cv.imshow("kp 2", image_2)
    # # cv.imshow("matches", img3)

    # estimate_homography(image_1, image_2, matches, keypoints_1_np, keypoints_2_np, 1000, 3, 400)
