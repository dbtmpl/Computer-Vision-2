import numpy as np
import cv2 as cv
import glob


# Since we use Python, instead of VLFeat we perform feature matching with ORB and BFMatcher. The corresponding code is based on
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html


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
        sample_size = 8  # Need 8 unknowns to solve for fundamental matrix. More will only introduce more outliers
    elif model_estimator == "eight_point":
        sample_size = 4
    else:
        print("No estimation technique specified")
        return None

    for n in np.arange(N):
        print("Current Iteration", n)
        sample = np.random.choice(matches, sample_size, replace=False)
        # F = fit_model_to_sample(sample, keypoints_1, keypoints_2)
        sample_kp_im1, sample_kp_im2 = get_matching_points(sample, keypoints_1, keypoints_2)
        if model_estimator == "baseline_homography":
            F = fit_model_to_sample_own_rationale(sample_kp_im1, sample_kp_im2)
        elif model_estimator == "eight_point":
            F = fit_model_to_sample_eight_point(sample_kp_im1, sample_kp_im2)
        else:
            print("Wrong estimator?")

        # Get points for testing F
        points, points_ = get_matching_points(matches, make_homogeneous(keypoints_1), make_homogeneous(keypoints_2))

        inliers = estimate_inliers(F, points, points_, t)
        number_inliers = inliers.shape[0]
        print("Number Inliers", number_inliers)

        if number_inliers > best_fit:
            best_fit = number_inliers

            if model_estimator == "baseline_homography":
                F = fit_model_to_sample_own_rationale(points[inliers, :2], points_[inliers, :2])
            elif model_estimator == "eight_point":
                F = fit_model_to_sample_eight_point(points[inliers, :2], points_[inliers, :2])
            else:
                print("Wrong estimator?")
                return None

            pp_with_inliers = (F @ points.T).T
            best_model = F

            # if number_inliers > good_enough:
            #     return best_model, points, points_, pp_with_inliers

    print("Best number of inliers:", best_fit)
    return best_model, points, points_, pp_with_inliers


def estimate_inliers(F, points, points_, t):
    predicted_points = (F @ points.T).T

    inliers = []
    for i, p_point in enumerate(predicted_points):
        gt_point = points_[i]

        diff = np.absolute(p_point - gt_point)

        if diff[0] < t and diff[1] < t:
            inliers.append(i)

    return np.asarray(inliers)


def make_homogeneous(array):
    return np.hstack((array, np.ones((array.shape[0], 1))))


def fit_model_to_sample_eight_point(keypoints_1, keypoints_2):
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


def fit_model_to_sample_own_rationale(keypoints_1, keypoints_2):
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


def get_matching_points(matches, keypoints_1, keypoints_2):
    image1_indices = [match.queryIdx for match in matches]
    image2_indices = [match.trainIdx for match in matches]

    return keypoints_1[image1_indices], keypoints_2[image2_indices]


if __name__ == "__main__":

    t = 2
    good_enough = 100

    image_data = [cv.imread(image) for image in sorted(glob.glob("Data/House/*.png"))]
    image_1 = image_data[0]
    image_2 = image_data[5]

    matches, kp_des_1, kp_des_2 = find_matches(image_1, image_2)

    # Convert opencv keypoint representation in numpy array
    keypoints_1_np = cv.KeyPoint.convert(kp_des_1[0])
    keypoints_2_np = cv.KeyPoint.convert(kp_des_2[0])

    best_model, points, points_, pp_with_inliers = Ransac(1000, matches, keypoints_1_np, keypoints_2_np, t,
                                                          good_enough, model_estimator="baseline_homography")

    pp_with_inliers = (best_model @ make_homogeneous(keypoints_1_np).T).T

    img3 = cv.drawMatches(image_1, kp_des_1[0], image_2, kp_des_2[0], matches, None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    for k in np.arange(points.shape[0]):
        cv.circle(image_1, (int(keypoints_1_np[k][0]), int(keypoints_1_np[k][1])), 2, (0, 255, 0), -1)
        cv.circle(image_2, (int(keypoints_2_np[k][0]), int(keypoints_2_np[k][1])), 4, (0, 255, 0), -1)
        cv.circle(image_2, (int(pp_with_inliers[k][0]), int(pp_with_inliers[k][1])), 2, (0, 0, 255), -1)

    while cv.waitKey(30):
        cv.imshow("kp 1", image_1)
        cv.imshow("kp 2", image_2)
        cv.imshow("matches", img3)
