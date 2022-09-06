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

from scipy.ndimage import maximum_filter
from scipy.ndimage import convolve

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--DataPath', default="../Data/Train/Set1", help='Path to the dataset folder to stitch')
    # TODO: tune these thresholds
    Parser.add_argument('--MatchRatioThreshold', default=1)
    Parser.add_argument('--TauThreshold', default=1)
    Parser.add_argument('--RansacMaxIterations', default=1)

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    DataPath = Args.DataPath
    ratio_threshold = Args.MatchRatioThreshold
    tau_threshold = Args.TauThreshold
    ransac_N_max = Args.RansacMaxIterations

    """
    Read a set of images for Panorama stitching
    """
    img_paths = glob.glob(os.path.join(DataPath, "*.jpg"))
    print(img_paths)
    imgs = [cv2.imread(path) for path in img_paths]
    img1 = imgs[0]
    img2 = imgs[1]
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    def get_corners(gray, img=None):
        ### corner harris
        corner_score = cv2.cornerHarris(gray, 4, 5, 0.04)

        #### goodFeaturesToTrack
        # corners = cv2.goodFeaturesToTrack(gray,1000,0.1,10)
        # corners = np.int0(corners)
        # print(corners.shape, gray.shape, corners)
        
        # Draw the corners
        if img != None:
            corners_img_viz = img
            corners_img_viz[corner_score>0.005*corner_score.max()]=[0,0,255]
            # cv2.imshow("corners", corners_img_viz)
            # cv2.waitKey()
            # cv2.imwrite("corners.png", corners_img_viz)
        return corner_score

    # get_corners(gray)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    corner_img: img with corner scores
    n_best: number of best corners
    returns n_best corners

    Save ANMS output as anms.png
    """
    def anms(corners_img, n_best, img=None):
        # get imregionalmax  res
        regional_max = maximum_filter(corners_img, size=50)
        regional_max_mask = (corners_img == regional_max).astype(np.uint8)
        
        kernel = np.ones((3, 3))
        new_maxima = convolve(regional_max_mask, kernel)
        new_maxima = np.where(new_maxima > 1, 0, new_maxima)

        new_maxima = new_maxima * regional_max_mask        
        new_maxima[new_maxima==1] = 255
        n_strong = np.where(new_maxima == 255)

        #get n best
        r_vec = []
        X,Y = n_strong[0],n_strong[1]
        
        for i in range(len(X)):
            r = np.inf
            for j in range(len(Y)):
                ed = None
                if(corners_img[X[j],Y[j]] > corners_img[X[i],Y[i]]):
                    ed = (X[j] - X[i])**2 + (Y[j] - Y[i])**2
                if(ed is not None and ed<r):
                    r=ed
            r_vec.append((i,r))
        n_best_index_vec = sorted(r_vec, key=lambda x: x[1], reverse=True)

        n_best_vec = []
        for i in range(min(n_best,len(n_best_index_vec))):
            n_best_vec.append([X[n_best_index_vec[i][0]], Y[n_best_index_vec[i][0]]])

        if img != None:
            anms_vis = img
            for pt in n_best_vec:
                cv2.drawMarker(anms_vis, (pt[1],pt[0]),  [0, 255, 0], cv2.MARKER_CROSS, 10, 1)
            # cv2.imshow("anms", anms_vis)
            # cv2.waitKey()
            cv2.imwrite("anms.png", anms_vis)
        
        return n_best_vec
    # n_best = anms(corner_score, 200)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    # Generate feature descriptors from around 41x41 patch around each feature
    # Gaussian blur (cv2.GaussianBlur(...)) for each feature patch
    # Sub-Sample the blurred patch to be 8x8, reshape to 64x1 vector
    # Standardize vector to zero mean and variance 1 (to remove bias)
    def get_feature_desc(n_best, gray):
        kernel_size = 40
        feature_vec = []
        for pt in n_best:
            left = pt[1] - kernel_size // 2
            x_min = max(0, left)
            right = pt[1] + kernel_size // 2
            x_max = min(gray.shape[1], right)
            top = pt[0] - kernel_size // 2
            y_min = max(0, top)
            bottom = pt[0] + kernel_size // 2
            y_max = min(gray.shape[0], bottom)

            patch_og = gray[y_min:y_max, x_min:x_max]
            patch = patch_og.copy()
            patch = np.hstack([np.tile(patch[:, :1], (1, x_min - left)),
                            patch,
                            np.tile(patch[:, -1:], (1, max(0, right - x_max)))])
            patch = np.vstack([np.tile(patch[:1, :], (y_min - top, 1)),
                            patch,
                            np.tile(patch[-1:, :], (max(0, bottom - y_max), 1))])

            # patch = gray[pt[0]-kernel_size//2:pt[0]+kernel_size//2,pt[1]-kernel_size//2:pt[1]+kernel_size//2]
            #perform gaussian blur
            patch_blur = cv2.GaussianBlur(patch, (3,3), 3)
            
            #subsample this shit
            subsample_step = kernel_size//8
            subsample, subsample_rows = [], []
            for i in range(0,kernel_size,subsample_step):
                subsample_rows.append(patch_blur[i])
            subsample_rows = np.array(subsample_rows)
            for i in range(0,kernel_size,subsample_step):
                subsample.append(subsample_rows[:,i])
            
            subsample = np.array(subsample).flatten()
            subsample_std = (subsample - subsample.mean())/ subsample.std()

            feature_vec.append(subsample_std)
             

        # subsample_viz = cv2.resize(feature_vec[0], (200,200), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("feature",subsample_viz)
        # cv2.waitKey()
        # cv2.imwrite("FD.png", subsample_viz)
        return feature_vec
    
    # feature_vec = get_feature_desc(n_best, gray)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    # For each vector pair, compute a sum-squared-dist-error
    # Sort by smallest-to-largest error
    # Take the smallest error and divide by 2nd-smallest to get a confidence ratio
    # Throw out "matches" that have a ratio that exceeds a defined threshold
    def feature_wrapper(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corner_score = get_corners(gray)
        n_best = anms(corner_score, 200)
        feature_vec = get_feature_desc(n_best, gray)
        # print(len(feature_vec), len(feature_vec[0]))
        return feature_vec, n_best


    def get_feature_matches(feature_vec1, feature_vec2, ratio_threshold, n_best1, n_best2, img1=None, img2=None):
        feature_matches = []
        for i in range(len(feature_vec1)):
            #tuple = (index1, index2, ssd_val)
            ssd_best, ssd_second_best = (0,0,np.inf), (0,0,np.inf)
            for j in range(len(feature_vec2)):
                ssd = ((feature_vec1[i]-feature_vec2[j])**2).sum()
                if ssd<ssd_best[2]:
                    ssd_second_best = ssd_best
                    ssd_best = (i,j,ssd)
            if(ssd_best[2]/ssd_second_best[2]<ratio_threshold):
                feature_matches.append([ssd_best[0],ssd_best[1]])
        # print(feature_matches)

        # if img1 != None and img2 != None:
        matching_img = np.hstack([img1, img2])
        n_best1 = np.array(n_best1).astype(float)
        n_best2 = np.array(n_best2).astype(float)
        matching_img = cv2.drawMatches(img1, [cv2.KeyPoint(y, x, 3) for x, y in n_best1.astype(float)],
                        img2, [cv2.KeyPoint(y, x, 3) for x, y in n_best2.astype(float)],
                        [cv2.DMatch(m1, m2, 0) for m1, m2 in feature_matches], matching_img, (0, 255, 0), (0, 0, 255))
        # cv2.imshow("matching",matching_img)
        # cv2.waitKey()
        # cv2.imwrite("matching.png", matching_img)
        
        
    vec1, n_best1 = feature_wrapper(img1)
    vec2, n_best2 = feature_wrapper(img2)
    get_feature_matches(vec1, vec2, 0.75, n_best1, n_best2, img1, img2)

    """
    Refine: RANSAC, Estimate Homography
    """
    # RANSAC:
    # 1) Select 4 randomly-selected feature pairs, (p_i1 from img_1, p_i2 from img_2) for i = [0,4)
    # 2) Compute homography between the two sets of points
    # 3) Compute inliers where SSD(p_i2, H_p_i) < tau_threshold (SSD = sum-square-difference)
    # 4) Repeat steps 1-3 until N_max iterations (or found 90% of total pts as inliers)
    # 5) Keep largest set of inliers that was found in the above steps/loop
    # 6) Re-compute least-squares Homography estimate on all inliers
    def ransac()

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    # Count the matching features between each pair of images to determine if there is a match
    # Take the base image and warp the 2nd to match its perspective
    # Then combine these matrices based on the Homography position
    # Blend the edges between image pairs
    # cv2.imwrite('mypano.png', pano_img)



if __name__ == "__main__":
    main()
