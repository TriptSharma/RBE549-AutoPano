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
    img = imgs[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    ### corner harris
    corner_score = cv2.cornerHarris(gray, 4, 5, 0.04)

    #### goodFeaturesToTrack
    # corners = cv2.goodFeaturesToTrack(gray,1000,0.1,10)
    # corners = np.int0(corners)
    # print(corners.shape, gray.shape, corners)
    
    # Draw the corners
    # print(corners)
    corners_img_viz = img
    corners_img_viz[corner_score>0.01*corner_score.max()]=[0,0,255]

    corners = np.where(corner_score>0.01*corner_score.max())
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.drawMarker(corners_img_viz,(x,y),[0, 0, 255], cv2.MARKER_CROSS, 10, 1)
    
    cv2.imshow("corners", corners_img_viz)
    cv2.waitKey()
    # cv2.imwrite("corners.png", corners_img_viz)


    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    def anms(corners_img, n_best):
        """
        corner_img: img with corner scores
        n_best: number of best corners
        returns n_best corners
        """
        # get imregionalmax  res
        regional_max = maximum_filter(corners_img, size=50)
        regional_max_mask = (corners_img == regional_max).astype(np.uint8)

        kernel = np.ones((3, 3))
        new_maxima = convolve(regional_max_mask, kernel)
        new_maxima = np.where(new_maxima > 1, 0, new_maxima)

        print(new_maxima)

        new_maxima = new_maxima * regional_max_mask

        new_maxima[new_maxima==1] = 255
        
        # cv2.imshow("regmax", new_maxima)
        # cv2.waitKey()

        n_strong = np.where(new_maxima == 255)

        #get n best
        r_vec = []
        X,Y = n_strong[0],n_strong[1]
        print(len(X))
        
        for i in range(len(X)):
            r = np.inf
            # print(n_strong[i],n_strong[i][0] )
            for j in range(len(Y)):
                ed = 0
                if(corners_img[X[j],Y[j]] > corners_img[X[i],Y[i]]):
                    ed = (X[j] - X[i])**2 + (Y[j] - Y[i])**2
                    # print(ed)
                if(ed<r):
                    r=ed
            if(r==np.inf):
                r=0
            r_vec.append((i,r))
            # print(i)
        n_best_index_vec = sorted(r_vec, key=lambda x: x[1], reverse=True)
        # print(n_best_index_vec[:n_best])
        n_best_vec = []
        for i in range(min(n_best,len(n_best_index_vec))):
            n_best_vec.append([X[n_best_index_vec[i][0]], Y[n_best_index_vec[i][0]]])
        print(n_best_vec)

        anms_vis = img
        for i in range(len(n_best_vec)):
            cv2.drawMarker(anms_vis, n_best_vec[i],  [0, 255, 0], cv2.MARKER_CROSS, 10, 1)

        cv2.imshow("anms", anms_vis)
        cv2.waitKey()
        
        return n_best_vec


    anms(corner_score, 200)
    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    # Generate feature descriptors from around 41x41 patch around each feature
    # Gaussian blur (cv2.GaussianBlur(...)) for each feature patch
    # Sub-Sample the blurred patch to be 8x8, reshape to 64x1 vector
    # Standardize vector to zero mean and variance 1 (to remove bias)
    # cv2.imwrite("FD.png", FD_img)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    # For each vector pair, compute a sum-squared-dist-error
    # Sort by smallest-to-largest error
    # Take the smallest error and divide by 2nd-smallest to get a confidence ratio
    # Throw out "matches" that have a ratio that exceeds a defined threshold
    # matching_img = cv2.drawMatches(...)
    # cv2.imwrite("matching.png", matching_img)

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
