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
from scipy.ndimage import maximum_filter, convolve

Parser = argparse.ArgumentParser()
Parser.add_argument('--NumFeatures', default=1000,
                    help='Number of best features to extract from each image, Default:1000')
Parser.add_argument('--DataPath', default="../Data/Test/TestSet2", help='Path to the dataset folder to stitch')
# TODO: tune these thresholds
Parser.add_argument('--MatchRatioThreshold', default=.75)
Parser.add_argument('--TauThreshold', default=160)
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
img_paths = sorted(glob.glob(os.path.join(DataPath, "*.jpg")))
imgs = [cv2.imread(path) for path in img_paths]
# img = imgs[2]
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

"""
Corner Detection
Save Corner detection output as corners.png
"""


def find_corners(img_gray, img=None):
    corner_score = cv2.cornerHarris(img_gray, 4, 5, 0.04)

    #### goodFeaturesToTrack
    # corners = cv2.goodFeaturesToTrack(gray,1000,0.1,10)
    # corners = np.int0(corners)
    # print(corners.shape, gray.shape, corners)

    # Draw the corners
    if img is not None:
        corners_img_viz = img.copy()
        corners_img_viz[corner_score > 0.005 * corner_score.max()] = [0, 0, 255]
        # cv2.imshow("corners", corners_img_viz)
        # cv2.waitKey()
        cv2.imwrite("corners.png", corners_img_viz)
    return corner_score


"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""


def anms(corners, img=None):
    # Implement ANMS algorithm
    # get imregionalmax  res
    regional_max = maximum_filter(corners, size=50)
    regional_max_mask = (corners == regional_max).astype(np.uint8)

    kernel = np.ones((3, 3))
    new_maxima = convolve(regional_max_mask, kernel)
    new_maxima[new_maxima > 1] = 0
    # new_maxima = np.where(new_maxima > 1, 0, new_maxima)

    new_maxima = new_maxima * regional_max_mask
    # new_maxima *= 255
    n_strong = np.where(new_maxima)

    # get n best
    r_vec = []
    X, Y = n_strong[0], n_strong[1]

    for i in range(len(X)):
        r = np.inf
        for j in range(len(Y)):
            ed = None
            if (corners[X[j], Y[j]] > corners[X[i], Y[i]]):
                ed = (X[j] - X[i]) ** 2 + (Y[j] - Y[i]) ** 2
            if (ed is not None and ed < r):
                r = ed
        r_vec.append((i, r))
    n_best_index_vec = sorted(r_vec, key=lambda x: x[1], reverse=True)

    n_best_vec = []
    for i in range(min(N_best, len(n_best_index_vec))):
        n_best_vec.append([Y[n_best_index_vec[i][0]], X[n_best_index_vec[i][0]]])

    if img is not None:
        anms_vis = img.copy()
        for pt in n_best_vec:
            cv2.drawMarker(anms_vis, (pt[0], pt[1]), [0, 255, 0], cv2.MARKER_CROSS, 10, 1)
        # cv2.imshow("anms", anms_vis)
        # cv2.waitKey()
        cv2.imwrite("anms.png", anms_vis)

    return np.array(n_best_vec)


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

        patch_gaussian = cv2.GaussianBlur(patch, (5, 5), 3)
        sub_sample_size = (8, 8)
        x_range = np.linspace(sub_sample_size[0] // 2, PatchSize - sub_sample_size[0] // 2, sub_sample_size[0],
                              dtype=int)
        y_range = np.linspace(sub_sample_size[0] // 2, PatchSize - sub_sample_size[0] // 2, sub_sample_size[0],
                              dtype=int)
        sub_sample = patch_gaussian[y_range]
        sub_sample = sub_sample[:, x_range]

        sub_sample2 = cv2.resize(patch_gaussian, sub_sample_size)
        # TODO: allowed to use this? for better consistency?
        # sub_sample = sub_sample2

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
        # print(np.sort(ssd))
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
            # print("failed to compute H")
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

    if img1 is not None:
        matches_img = np.hstack([img1, img2])
        out = cv2.drawMatches(img1, [cv2.KeyPoint(x, y, 3) for x, y in corners1_matched.astype(float)],
                              img2, [cv2.KeyPoint(x, y, 3) for x, y in corners2_matched.astype(float)],
                              [cv2.DMatch(m1, m1, 0) for m1 in max_inliers_list], matches_img, (0, 255, 0), (0, 0, 255))
        cv2.imwrite("ransac.png", out)
    max_inliers_H = cv2.findHomography(corners1_matched[max_inliers_list], corners2_matched[max_inliers_list], 0)[0]

    return max_inliers_H, max_inliers_list


"""
Image Warping + Blending
Save Panorama output as mypano.png
"""
# Count the matching features between each pair of images to determine if there is a match
# Take the base image and warp the 2nd to match its perspective
# Then combine these matrices based on the Homography position
# Blend the edges between image pairs
# cv2.imwrite('mypano.png', pano_img)
def stitch(img1, img2, H1, H2):
    # H1 = np.eye(3)
    # H1[0, -1] += 0
    # H1[1, -1] += 0
    t = [600, 600]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]).astype(float)  # translate

    # H2 = np.linalg.inv(H2)
    # H2[0, -1] += 0
    # H2[1, -1] += 0
    H1 = Ht.dot(H1)
    H2 = Ht.dot(H2)

    img1_warp = cv2.warpPerspective(img1, H1, (img1.shape[1] * 4, img1.shape[0] * 4))
    img2_warp = cv2.warpPerspective(img2, H2, (img2.shape[1] * 4, img2.shape[0] * 4))
    # img_combo = cv2.bitwise_or(img1, img2)
    overlap_mask = np.any(img1_warp > 0, axis=2) & np.any(img2_warp > 0, axis=2)
    img_combo = (img1_warp * ~np.atleast_3d(overlap_mask)).astype(float) + img2_warp.astype(float)
    return img_combo


corners = [find_corners(img_gray, img) for img_gray, img in zip(imgs_gray, imgs)]
best_corners = [anms(cs, img) for cs, img in zip(corners, imgs)]
corners_descriptors = [get_descriptors(img_gray, corners_best) for img_gray, corners_best in
                       zip(imgs_gray, best_corners)]

def check_and_stitch(i, j):
    img1 = imgs[i]
    img2 = imgs[j]

    img0_1_matches = match(corners_descriptors[i], corners_descriptors[j], best_corners[i], best_corners[j], img1, img2)

    if len(img0_1_matches) < 20:
        return None

    ransac_H_out, inliers = ransac(best_corners[i], best_corners[j], img0_1_matches, img1, img2)
    # print("det", np.linalg.det(ransac_H_out))
    if np.linalg.det(ransac_H_out) < .3:
        return None
    out = stitch(img1, img2, ransac_H_out)
    return out

MIN_MATCHES_RATIO = .25

connectivity_matrix = np.zeros((len(imgs_gray), len(imgs_gray)))
connectivity_transforms = np.full((len(imgs_gray), len(imgs_gray), 3, 3), np.eye(3))

pairs = {}
for i, img_gray_i in enumerate(imgs_gray):
    for j, img_gray_j in enumerate(imgs_gray):
        if i == j:# or (i, j) in pairs.keys() or (j, i) in pairs.keys():
            continue
        matches = match(corners_descriptors[i], corners_descriptors[j], best_corners[i], best_corners[j], imgs[i], imgs[j])
        if (len(matches) / ((len(best_corners[i]) + len(best_corners[j]))/2)) < .2:
            continue
        ransac_H, inliers = ransac(best_corners[i], best_corners[j], matches, img_gray_i, img_gray_j)
        # print('det', (i, j), np.linalg.det(ransac_H))
        if ransac_H is None or np.linalg.det(ransac_H) < .3:
            # print("Bad transform", np.linalg.det(ransac_H))
            continue
        match_score = (len(inliers) / len(matches))
        if match_score > MIN_MATCHES_RATIO:
            # print("match", (i, j), match_score)
            pairs[(i, j)] = match_score
            connectivity_matrix[i, j] = len(inliers)
            # connectivity_matrix[j, i] = match_score
            connectivity_transforms[i, j, :, :] = ransac_H
            # connectivity_transforms[j, i, :, :] = ransac_H
        # else:
            # print("no match", (i, j), match_score)


#### REMOVE

visited = np.zeros((len(imgs_gray), len(imgs_gray)))
weights = connectivity_matrix

graph = {}
for r in range(len(weights)):
    lst = []
    for c in range(len(weights[r])):
        if weights[r, c] > 0:
            lst.append(c)
    graph[r] = lst

visited = set() # Set to keep track of visited nodes of graph.

paths = []
def dfs(visited, graph, node, path0, last_node, path_weight):  #function for dfs
    path = path0.copy()
    if node not in visited:
        # print(node)
        path.append((last_node, node))
        path_weight += weights[last_node, node]

        visited.add(node)

        for neighbour in graph[node]:
            dfs(visited, graph, neighbour, path, node, path_weight)
    else:
        paths.append((path, path_weight))


for start_r in range(len(imgs_gray)):
    visited = set()
    dfs(visited, graph, 0, [], 0, 0)


# print("paths", paths)

longest_path, max_score = max(paths, key=lambda x: len(x[0]))

center = longest_path[len(longest_path)//2][0]

longest_path = np.array(longest_path)
### REMOVE



# for row in range(len(imgs_gray)):
#
# paths = []
# def recurse(r, c, path, path_weight):
#     # paths = []
#     # for col in range(len(imgs_gray)):
#     if r < 0 or r >= visited.shape[0] or c < 0 or c >= visited.shape[1] or visited[r, c]:
#         return path, path_weight
#
#     if r == c:
#         paths.append(path)
#         return path, path_weight
#     if weights[r, c] < 1:
#         paths.append(path)
#         return path, path_weight
#     if visited[r, c]:
#         paths.append(path)
#         return path, path_weight
#     visited[r, c] = 1
#     visited[c, r] = 1  # TODO: maybe?
#
#     path.append((r, c))
#     path_weight += weights[r, c]
#
#     [recurse(c, 0, path, path_weight) for c in range(len(imgs_gray))]
#
#
#
#
#
#
#
# best_core_img = 0
# best_score = 0
# for i in range(connectivity_matrix.shape[0]):
#     if np.count_nonzero(connectivity_matrix[i, :]) < 2:
#         continue
#     score = np.sum(connectivity_matrix[i, :])
#     if score > best_score:
#         best_core_img = i
#         best_score = score
#
# known_imgs = [best_core_img]
# pairs = [(best_core_img, best_core_img)]
# unknown_imgs = set(range(connectivity_matrix.shape[1]))
# unknown_imgs.remove(best_core_img)
#
#
#
# while len(unknown_imgs) > 0:
#     for j in unknown_imgs.copy():
#         if np.count_nonzero(connectivity_matrix[:, j]) < 1 and j in unknown_imgs:
#             unknown_imgs.remove(j)
#             continue
#         if np.count_nonzero(connectivity_matrix[known_imgs, j]) < 1 or j in [p[0] for p in pairs]:
#             continue
#         best_starter = np.argsort(connectivity_matrix[known_imgs, j])[-1]
#         best_starter = known_imgs[best_starter]
#         print(best_starter, "->", j)
#         known_imgs.append(j)
#         unknown_imgs.remove(j)
#         pairs.append((best_starter, j))
#
# running_H = np.eye(3)
# for j in range(connectivity_matrix.shape[1]):
#     if j == best_core_img:
#         continue

combos = []
last_H = np.eye(3)

# last_H = Ht.dot(np.eye(3))
# center = 0

# for i in range(center, len(imgs_gray)-1):
# for i, j in longest_path[center:]:
#     # j = i+1
#
#     # H2[0, -1] += 0
#     # H2[1, -1] += 0
#     # H1 = Ht.dot(H1)
#     # H2 = Ht.dot(H2)
#
#     next_H = np.linalg.inv(connectivity_transforms[i, j]).dot(last_H)
#
#     combos.append(stitch(imgs[i], imgs[j], last_H, next_H))
#     last_H = next_H
#
# last_H = np.eye(3)
# for i, j in reversed(longest_path[:center]):
#     # j = i-1
#     next_H = np.linalg.inv(connectivity_transforms[i, j]).dot(last_H)
#
#     combos.insert(0, stitch(imgs[i], imgs[j], last_H, next_H))
#     last_H = next_H


for i, j in longest_path:
    # j = i+1

    # H2[0, -1] += 0
    # H2[1, -1] += 0
    # H1 = Ht.dot(H1)
    # H2 = Ht.dot(H2)

    next_H = np.linalg.inv(connectivity_transforms[i, j]).dot(last_H)

    combos.append(stitch(imgs[i], imgs[j], last_H, next_H))
    last_H = next_H



# img_combo01 = check_and_stitch(0, 3)
# cv2.imwrite("mypano_01.png", img_combo01)

def merge(img1, img2):
    base_avg_img = (img1 + img2) / 2

    # overlap_mask = np.any(img1 > 0, axis=2) & np.any(img2 > 0, axis=2)
    # img_combo = (img1 * ~np.atleast_3d(overlap_mask)).astype(float) + img2.astype(float)

    overlap_mask = np.any(img1 > 0, axis=2) & np.any(img2 > 0, axis=2)
    part1 = (img1 * ~np.atleast_3d(overlap_mask)).astype(float)
    part2 = img2.astype(float)
    base_avg_img[part1 > 1] = part1[part1 > 1]
    base_avg_img[part2 > 1] = part2[part2 > 1]

    return base_avg_img

out = combos[center]
for i in range(center+1, len(combos)):
    out = merge(out, combos[i])

for i in range(center-1, -1, -1):
    out = merge(combos[i], out)

# img_combo12 = check_and_stitch(0, 2)
cv2.imwrite("mypano_full.png", out[200:2100, :])

print('a')

# if __name__ == "__main__":
#     main()
print('b')
