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

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--DataPath', default="./Data/Train/Set1", help='Path to the dataset folder to stitch')
    # TODO: tune these thresholds
    Parser.add_argument('--MatchRatioThreshold', default=1)
    Parser.add_argument('--TauThreshold', default=1)
    Parser.add_argument('--RansacNMax', default=1)

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    DataPath = Args.DataPath
    ratio_threshold = Args.MatchRatioThreshold
    tau_threshold = Args.TauThreshold
    ransac_N_max = Args.RansacNMax

    """
    Read a set of images for Panorama stitching
    """
    img_paths = glob.glob(os.path.join(DataPath, "*.jpg"))
    imgs = [cv2.imread(path) for path in img_paths]
    img = imgs[0]

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    # corners = cv2.cornerHarris(img, ...)
    # Draw the corners
    # cv2.imwrite("corners.png", corners_img)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    # Implement ANMS algorithm
    # Initialize r_i = infinity for i = [1 : N_strong]
    # for i = [1 : N_strong] do
    #   for j = [1 : N_strong] do
    #       if (C_img(y_j, x_j) > C_img(y_i, x_i)) then
    #           ED = (x_j - x_i)^2 + (y_j - y_i)^2
    #       end
    #       if ED < r_i then
    #           r_i = ED
    #       end
    #   end
    # end
    # Sort r_i in descending order and pick top N_best points

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
