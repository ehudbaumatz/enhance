import numpy as np
import cv2 as cv

def draw_keypoints(img, method='AKAZE'):

    img = cv.imread(img) if type(img) == str else img
    if method == 'AKAZE':
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        akaze = cv.AKAZE_create()
        kp, descriptor = akaze.detectAndCompute(gray, None)

        img=cv.drawKeypoints(gray, kp, img)
        return img

def match_by_keypoints(img1, img2):
    """
    we need to associate, or “match”, keypoints from both images that correspond in reality to the same point.
    One possible method is BFMatcher.knnMatch(). This matcher measures the distance between each pair of keypoint descriptors and returns for each keypoint its k best matches with the minimal distance.
    :param img1:
    :param img2:
    :return:
    """

    img1 = cv.imread(img1, cv.IMREAD_GRAYSCALE) if type(img1) == str else img1   # referenceImage
    img2 = cv.imread(img2, cv.IMREAD_GRAYSCALE) if type(img2) == str else img2  # sensedImage

    # Initiate AKAZE detector
    akaze = cv.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Draw matches
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

    # Compute homography
    H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC, 5.0)

    # Warp image
    warped_image = cv.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

    return warped_image, img3, good_matches, matches

