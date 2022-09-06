#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 1 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import argparse
import glob
import os

# def main():
# Add any Command Line arguments here
from matplotlib import pyplot as plt
from usb._lookup import descriptors

Parser = argparse.ArgumentParser()
Parser.add_argument('--NumFeatures', default=1000,
                    help='Number of best features to extract from each image, Default:1000')
Parser.add_argument('--DataPath', default="../Data/Train/Set1", help='Path to the dataset folder to stitch')
# TODO: tune these thresholds
Parser.add_argument('--MatchRatioThreshold', default=.85)
Parser.add_argument('--TauThreshold', default=200)
Parser.add_argument('--RansacNMax', default=500)

Args = Parser.parse_args()
NumFeatures = Args.NumFeatures
DataPath = Args.DataPath
ratio_threshold = Args.MatchRatioThreshold
tau_threshold = Args.TauThreshold
ransac_N_max = Args.RansacNMax

N_corners = NumFeatures
N_best = 200
PatchSize = 41

"""
Read a set of images for Panorama stitching
"""
img_paths = glob.glob(os.path.join(DataPath, "*.jpg"))
imgs = [cv2.imread(path) for path in img_paths]
# img = imgs[2]
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

"""
Corner Detection
Save Corner detection output as corners.png
"""


def find_corners(img_gray, img=None):
    corners = cv2.goodFeaturesToTrack(img_gray, N_corners, .005, 5)
    corners = corners[:, 0, :].astype(int)

    # corners = cv2.cornerHarris(img_gray, 3, 3, .001)
    # imgOut = img.copy()
    # imgOut[corners > 0.005*corners.max()] = [0,0,255]

    # Draw the corners
    if img is not None:
        corners_img = img.copy()
        for x, y in corners:
            cv2.circle(corners_img, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite("corners.png", corners_img)
    return corners


"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""


def anms(corners, img=None):
    # Implement ANMS algorithm
    r = [np.inf for i in range(len(corners))]
    for i in range(len(corners)):
        for j in range(len(corners)):
            if j > i:
                ED = (corners[j, 0] - corners[i, 0]) ** 2 + (corners[j, 1] - corners[i, 1]) ** 2
                if ED < r[i]:
                    r[i] = ED

    # Sort r_i in descending order and pick top N_best points
    r_i_sorted = np.argsort(r)
    corners_best = corners[r_i_sorted[:N_best]]

    if img is not None:
        anms_img = img.copy()
        for x, y in corners_best:
            cv2.circle(anms_img, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite("anms.png", anms_img)
    return corners_best


"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""


def get_descriptors(img_gray, corners_best):
    # Generate feature descriptors from around 41x41 patch around each feature
    # Gaussian blur (cv2.GaussianBlur(...)) for each feature patch
    # Sub-Sample the blurred patch to be 8x8, reshape to 64x1 vector
    # Standardize vector to zero mean and variance 1 (to remove bias)
    # cv2.imwrite("FD.png", FD_img)
    feature_descriptors = []
    for x, y in corners_best:
        left = x - PatchSize // 2
        x_min = max(0, left)
        right = x + PatchSize // 2
        x_max = min(img_gray.shape[1], right)
        top = y - PatchSize // 2
        y_min = max(0, top)
        bottom = y + PatchSize // 2
        y_max = min(img_gray.shape[0], bottom)

        patch_og = img_gray[y_min:y_max, x_min:x_max]
        patch = patch_og.copy()
        patch = np.hstack([np.tile(patch[:, :1], (1, x_min - left)),
                           patch,
                           np.tile(patch[:, -1:], (1, max(0, right - x_max)))])
        patch = np.vstack([np.tile(patch[:1, :], (y_min - top, 1)),
                           patch,
                           np.tile(patch[-1:, :], (max(0, bottom - y_max), 1))])

        patch_gaussian = cv2.GaussianBlur(patch, (3, 3), 3)
        sub_sample_size = (8, 8)
        x_range = np.linspace(sub_sample_size[0] // 2, PatchSize - sub_sample_size[0] // 2, sub_sample_size[0],
                              dtype=int)
        y_range = np.linspace(sub_sample_size[0] // 2, PatchSize - sub_sample_size[0] // 2, sub_sample_size[0],
                              dtype=int)
        sub_sample = patch_gaussian[y_range]
        sub_sample = sub_sample[:, x_range]

        sub_sample2 = cv2.resize(patch_gaussian, sub_sample_size)
        # TODO: allowed to use this? for better consistency?
        sub_sample = sub_sample2

        desc_vector = sub_sample.reshape((-1,)).astype(float)
        variance = np.var(desc_vector)
        mean = np.mean(desc_vector)
        desc_vector_std = (desc_vector - mean) / np.sqrt(variance)
        feature_descriptors.append(desc_vector_std)

    cv2.imwrite('FD_img.png', sub_sample)
    return np.array(feature_descriptors)


"""
Feature Matching
Save Feature Matching output as matching.png
"""


def match(feature_descriptors_1, feature_descriptors_2, corners1=None, corners2=None, img1=None, img2=None):
    # For each vector pair, compute a sum-squared-dist-error
    # Sort by smallest-to-largest error
    # Take the smallest error and divide by 2nd-smallest to get a confidence ratio
    # Throw out "matches" that have a ratio that exceeds a defined threshold
    # matching_img = cv2.drawMatches(...)
    # cv2.imwrite("matching.png", matching_img)
    matches = []
    for i, desc_i in enumerate(feature_descriptors_1):
        diffs = feature_descriptors_2 - desc_i
        ssd = np.sum(np.square(diffs), axis=1)
        sorted_ssd_args = np.argsort(ssd)
        j = sorted_ssd_args[0]
        ratio = ssd[j] / ssd[sorted_ssd_args[1]]
        if ratio < ratio_threshold:
            matches.append((i, j))

    if corners1 is not None:
        matches_img = np.hstack([img1, img2])
        out = cv2.drawMatches(img1, [cv2.KeyPoint(x, y, 3) for x, y in corners1.astype(float)],
                              img2, [cv2.KeyPoint(x, y, 3) for x, y in corners2.astype(float)],
                              [cv2.DMatch(m1, m2, 0) for m1, m2 in matches], matches_img, (0, 255, 0), (0, 0, 255))
        cv2.imwrite("matching.png", out)
    return matches


"""
Refine: RANSAC, Estimate Homography
"""


def ransac(corners1, corners2, matches_ij, img1=None, img2=None):
    # RANSAC:
    # 1) Select 4 randomly-selected feature pairs, (p_i1 from img_1, p_i2 from img_2) for i = [0,4)
    # 2) Compute homography between the two sets of points
    # 3) Compute inliers where SSD(p_i2, H_p_i) < tau_threshold (SSD = sum-square-difference)
    # 4) Repeat steps 1-3 until N_max iterations (or found 90% of total pts as inliers)
    # 5) Keep largest set of inliers that was found in the above steps/loop
    # 6) Re-compute least-squares Homography estimate on all inliers
    matches_ij = np.array(matches_ij)
    corners1_matched = corners1[matches_ij[:, 0]]
    corners2_matched = corners2[matches_ij[:, 1]]
    max_inliers_list = []
    for n in range(ransac_N_max):
        # random_idx = np.random.randint(0, len(matches_ij)-1, 4)
        random_idxs = (np.random.random_sample((4)) * (len(matches_ij) - 1)).astype(int)
        matrix = np.zeros((len(random_idxs) * 2 + 1, len(random_idxs) * 2 + 1))
        for i, random_i in enumerate(random_idxs):
            # i1, i2 = matches_ij[random_i]
            x1, y1 = corners1_matched[random_i]
            x2, y2 = corners2_matched[random_i]
            matrix[i * 2, :] = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]
            matrix[i * 2 + 1, :] = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]
        matrix[-1, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        # H = np.linalg.inv(matrix.T * matrix) @ (matrix.T * vec)
        # H = np.linalg.lstsq(matrix, vec.T)[0]
        # H = np.concatenate((H, [[1]])).reshape((3, 3))

        # u, s, v = np.linalg.svd(matrix)
        # h = np.reshape(np.hstack((v[7], [1])), (3, 3))
        # H1 = (1 / h[2, 2]) * h
        try:
            v = np.zeros((9,))
            v[-1] = 1
            H1 = np.linalg.inv(matrix) @ v
            H1 = H1.reshape((3, 3))
        except:
            print("failed to compute H")
            continue

        # TODO: fix scratch homography
        H2 = cv2.findHomography(corners1_matched[random_idxs].reshape(-1, 1, 2),
                                corners2_matched[random_idxs].reshape(-1, 1, 2), 0)[0]
        # if H1 is None:
        #     continue

        corners1_H = cv2.perspectiveTransform(corners1_matched.reshape(-1, 1, 2).astype(float), H1)

        SSD = np.sum(np.square(corners1_H.reshape(-1, 2) - corners2_matched), axis=1)
        inlier_args = np.where(SSD < tau_threshold)[0]
        if len(inlier_args) > len(max_inliers_list):
            max_inliers_list = inlier_args
        # corners1_H = np.hstack([corners1, np.ones((len(corners1), 1))]) @ H2
        if len(max_inliers_list) > 0.9 * len(corners1):
            print("Over 90% inliers!")
            break

        print("a")
    if img1 is not None:
        matches_img = np.hstack([img1, img2])
        out = cv2.drawMatches(img1, [cv2.KeyPoint(x, y, 3) for x, y in corners1_matched.astype(float)],
                              img2, [cv2.KeyPoint(x, y, 3) for x, y in corners2_matched.astype(float)],
                              [cv2.DMatch(m1, m1, 0) for m1 in max_inliers_list], matches_img, (0, 255, 0), (0, 0, 255))
        cv2.imwrite("ransac.png", out)
    max_inliers_H = cv2.findHomography(corners1_matched[max_inliers_list], corners2_matched[max_inliers_list], 0)[0]

    return max_inliers_H


"""
Image Warping + Blending
Save Panorama output as mypano.png
"""
# Count the matching features between each pair of images to determine if there is a match
# Take the base image and warp the 2nd to match its perspective
# Then combine these matrices based on the Homography position
# Blend the edges between image pairs
# cv2.imwrite('mypano.png', pano_img)
def stitch(img1, img2, ransac_H_out):
    H1 = np.eye(3)
    # H1[0, -1] += 0
    # H1[1, -1] += 0
    t = [400, 400]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    H2 = np.linalg.inv(ransac_H_out)
    # H2[0, -1] += 0
    # H2[1, -1] += 0
    H1 = Ht.dot(H1)
    H2 = Ht.dot(H2)

    img1_warp = cv2.warpPerspective(img1, H1, (img1.shape[1] * 3, img1.shape[0] * 3))
    img2_warp = cv2.warpPerspective(img2, H2, (img2.shape[1] * 3, img2.shape[0] * 3))
    # img_combo = cv2.bitwise_or(img1, img2)
    overlap_mask = np.any(img1_warp > 0, axis=2) & np.any(img2_warp > 0, axis=2)
    img_combo = (img1_warp * ~np.atleast_3d(overlap_mask)).astype(float) + img2_warp.astype(float)
    return img_combo


corners = [find_corners(img_gray) for img_gray in imgs_gray]
best_corners = [anms(cs) for cs in corners]
corners_descriptors = [get_descriptors(img_gray, corners_best) for img_gray, corners_best in
                       zip(imgs_gray, best_corners)]

def check_and_stitch(i, j):
    img1 = imgs[i]
    img2 = imgs[j]

    img0_1_matches = match(corners_descriptors[i], corners_descriptors[j], best_corners[i], best_corners[j], img1, img2)

    if len(img0_1_matches) < 20:
        return None

    ransac_H_out = ransac(best_corners[i], best_corners[j], img0_1_matches, img1, img2)
    out = stitch(img1, img2, ransac_H_out)
    return out

img_combo01 = check_and_stitch(0, 1)
img_combo12 = check_and_stitch(0, 2)
overlap_mask = np.any(img_combo01 > 0, axis=2) & np.any(img_combo12 > 0, axis=2)
img_combo = (img_combo12 * ~np.atleast_3d(overlap_mask)).astype(float) + img_combo01.astype(float)
cv2.imwrite("mypano_full.png", img_combo)

print('a')

# if __name__ == "__main__":
#     main()
