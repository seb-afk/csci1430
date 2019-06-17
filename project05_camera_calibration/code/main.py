# Camera Calibration Stencil Code
# Transferred to python by Eleanor Tursman, based on previous work by Henry Hu, 
# Grady Williams, and James Hays for CSCI 1430 @ Brown and 
# CS 4495/6476 @ Georgia Tech
#
# This script 
# (1) Loads 2D and 3D data points and images
# (2) Calculates the projection matrix from those points    (you code this)
# (3) Computes the camera center from the projection matrix (you code this)
# (4) Estimates the fundamental matrix                      (you code this)
# (5) Draws the epipolar lines on images
# (6) Calls skimage's HOG matching function
# (7) Estimates the fundamental matrix using RANSAC 
#     and filters away spurious matches                     (you code this)
# (8) Draws the epipolar lines on images and corresponding matches
#
# The relationship between coordinates in the world and coordinates in the
# image defines the camera calibration. See Szeliski 6.2, 6.3 for reference.
#
# 2 pairs of corresponding points files are provided
# Ground truth is provided for pts2d-norm-pic_a and pts3d-norm pair
# You need to report the values calculated from pts2d-pic_b and pts3d

import numpy as np
import os 
import matplotlib.pyplot as plt
from skimage import io
from scipy import misc
import cv2
from student import (calculate_projection_matrix, compute_camera_center, 
estimate_fundamental_matrix, ransac_fundamental_matrix)
from helpers import (evaluate_points, visualize_points, plot3dview, 
draw_epipolar_lines, matchAndShowCorrespondence, showCorrespondence)

def main():

    data_dir = os.path.dirname(__file__) + '../data/'

    ########## Parts (1) through (3)
    Points_2D = np.loadtxt(data_dir + 'pts2d-norm-pic_a.txt')
    Points_3D = np.loadtxt(data_dir + 'pts3d-norm.txt')

    # (Optional) Uncomment these two lines once you have your code working
    # with the easier, normalized points above.
    #Points_2D = np.loadtxt(data_dir + 'pts2d-pic_b.txt')
    #Points_3D = np.loadtxt(data_dir + 'pts3d.txt')

    # Calculate the projection matrix given corresponding 2D and 3D points
    # !!! You will need to implement calculate_projection_matrix. !!!
    M = calculate_projection_matrix(Points_2D,Points_3D)
    print('The projection matrix is:\n {0}\n'.format(M))

    Projected_2D_Pts, Residual = evaluate_points(M, Points_2D, Points_3D)
    print('The total residual is:\n {0}\n'.format(Residual))

    visualize_points(Points_2D,Projected_2D_Pts)

    # Calculate the camera center using the M found from previous step
    # !!! You will need to implement compute_camera_center. !!!
    Center = compute_camera_center(M)
    print('The estimated location of the camera is:\n {0}\n'.format(Center))

    plot3dview(Points_3D, Center)

    ########## Parts (4) and (5)
    Points_2D_pic_a = np.loadtxt(data_dir + 'pts2d-pic_a.txt')
    Points_2D_pic_b = np.loadtxt(data_dir + 'pts2d-pic_b.txt')

    ImgLeft  = io.imread(data_dir + 'pic_a.jpg')
    ImgRight = io.imread(data_dir + 'pic_b.jpg')

    # (Optional) You might try adding noise for testing purposes:
    #Points_2D_pic_a = Points_2D_pic_a + 6*np.random.rand(Points_2D_pic_a.shape[0],Points_2D_pic_a.shape[1])-0.5
    #Points_2D_pic_b = Points_2D_pic_b + 6*np.random.rand(Points_2D_pic_b.shape[0],Points_2D_pic_b.shape[1])-0.5

    # Calculate the fundamental matrix given corresponding point pairs
    # !!! You will need to implement estimate_fundamental_matrix. !!!
    F_matrix = estimate_fundamental_matrix(Points_2D_pic_a,Points_2D_pic_b)

    # Draw the epipolar lines on the images
    draw_epipolar_lines(F_matrix,ImgLeft,ImgRight,Points_2D_pic_a,Points_2D_pic_b)

    ########## Parts (6) through (8)
    # This Mount Rushmore pair is easy. Most of the initial matches are
    # correct. The base fundamental matrix estimation without coordinate
    # normalization will work fine with RANSAC.
    pic_a = io.imread(data_dir + 'Mount Rushmore/9193029855_2c85a50e91_o.jpg')
    pic_b = io.imread(data_dir + 'Mount Rushmore/7433804322_06c5620f13_o.jpg')
    pic_a = misc.imresize(pic_a,0.25,interp='bilinear')
    pic_b = misc.imresize(pic_b,0.37,interp='bilinear')

    # The Notre Dame pair is difficult because the keypoints are largely on the
    # same plane. Still, even an inaccurate fundamental matrix can do a pretty
    # good job of filtering spurious matches.
    # pic_a = io.imread(data_dir + 'Notre Dame/921919841_a30df938f2_o.jpg')
    # pic_b = io.imread(data_dir + 'Notre Dame/4191453057_c86028ce1f_o.jpg')
    # pic_a = misc.imresize(pic_a,0.5,interp='bilinear')
    # pic_b = misc.imresize(pic_b,0.5,interp='bilinear')

    # The Gaudi pair doesn't find many correct matches unless you run at high
    # resolution, but that will lead to tens of thousands of ORB features
    # which will be somewhat slow to process. Normalizing the coordinates
    # (extra credit) seems to make this pair work much better.
    # pic_a = io.imread(data_dir + 'Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg')
    # pic_b = io.imread(data_dir + 'Episcopal Gaudi/4386465943_8cf9776378_o.jpg')
    # pic_a = misc.imresize(pic_a,0.8,interp='bilinear')
    # pic_b = misc.imresize(pic_b,1.0,interp='bilinear')

    # This pair of photos has a clearer relationship between the cameras (they
    # are converging and have a wide baseine between them) so the estimated
    # fundamental matrix is less ambiguous and you should get epipolar lines
    # qualitatively similar to part 2 of the project.
    # pic_a = io.imread(data_dir + 'Woodruff Dorm/wood1.jpg')
    # pic_b = io.imread(data_dir + 'Woodruff Dorm/wood2.jpg')
    # pic_a = misc.imresize(pic_a,0.65,interp='bilinear')
    # pic_b = misc.imresize(pic_b,0.65,interp='bilinear')

    # Finds matching points in the two images using opencv's implementation of
    # ORB. There can still be many spurious matches, though.
    [Points_2D_pic_a,Points_2D_pic_b] = matchAndShowCorrespondence(pic_a,pic_b)
    print('Found {0} possibly matching features\n'.format(Points_2D_pic_a.shape[0]))

    # Calculate the fundamental matrix using RANSAC
    # !!! You will need to implement ransac_fundamental_matrix. !!!
    [F_matrix,matched_points_a,matched_points_b] = ransac_fundamental_matrix(Points_2D_pic_a,Points_2D_pic_b)

    # Draw the epipolar lines on the images and corresponding matches
    showCorrespondence(pic_a, pic_b, matched_points_a, matched_points_b)
    draw_epipolar_lines(F_matrix,pic_a,pic_b,matched_points_a,matched_points_b)

    # optional - re estimate the fundamental matrix using ALL the inliers.
    #F_matrix = estimate_fundamental_matrix(matched_points_a,matched_points_b)
    #draw_epipolar_lines(F_matrix,pic_a,pic_b,matched_points_a,matched_points_b)


if __name__ == '__main__':
    main()
