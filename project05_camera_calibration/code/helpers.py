# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu, 
# Grady Williams, and James Hays for CSCI 1430 @ Brown and 
# CS 4495/6476 @ Georgia Tech

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from matplotlib.patches import ConnectionPatch
from skimage import io,color,feature,img_as_float32
from mpl_toolkits.mplot3d import Axes3D


# Visualize the actual 2D points and the projected 2D points calculated
# from the projection matrix
# You do not need to modify anything in this function, although you can if
# you want to.
def evaluate_points(M, Points_2D, Points_3D):

    reshaped_points = np.concatenate(
                      (Points_3D,np.ones((Points_3D.shape[0],1))),
                      axis=1)
    Projection = np.matmul(M,np.transpose(reshaped_points))
    Projection = np.transpose(Projection)
    u = np.divide(Projection[:,0],Projection[:,2])
    v = np.divide(Projection[:,1],Projection[:,2])
    Residual = np.sum(np.power(np.power(u-Points_2D[:,0],2) + np.power(v-Points_2D[:,1],2),0.5))
    Projected_2D_Pts = np.transpose(np.vstack([u,v]))

    return Projected_2D_Pts, Residual

# Visualize the actual 2D points and the projected 2D points calculated
# from the projection matrix
# You do not need to modify anything in this function, although you can if
# you want to.
def visualize_points(Actual_Pts, Project_Pts):
    plt.scatter(Actual_Pts[:,0],Actual_Pts[:,1],marker='o')
    plt.scatter(Project_Pts[:,0],Project_Pts[:,1],marker='x')
    plt.legend(('Actual Points','Projected Points'))
    plt.show()

# Visualize the actual 3D points and the estimated 3D camera center.
# You do not need to modify anything in this function, although you can if
# you want to.
def plot3dview(Points_3D, camera_center1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Points_3D[:,0],Points_3D[:,1],Points_3D[:,2],c='b',marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.elev = 31
    ax.azim = -129

    # draw vertical lines connecting each point to Z=0
    min_z = np.min(Points_3D[:,2])
    for i in range(0,Points_3D.shape[0]):
       ax.plot(np.array([Points_3D[i,0],Points_3D[i,0]]), 
               np.array([Points_3D[i,1],Points_3D[i,1]]),
               np.array([Points_3D[i,2],min_z]))

    # if camera_center1 exists, plot it
    if 'camera_center1' in locals():
        ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],s=100,c='r',marker='x')
        ax.plot(np.array([camera_center1[0],camera_center1[0]]),
                np.array([camera_center1[1],camera_center1[1]]),
                np.array([camera_center1[2],min_z]),c='r')

    plt.show()

# Draw the epipolar lines given the fundamental matrix, left right images
# and left right datapoints
# You do not need to modify anything in this function, although you can if
# you want to.
def draw_epipolar_lines(F_matrix, ImgLeft, ImgRight, PtsLeft, PtsRight):
    
    Pul = np.array([1,1,1])
    Pbl = np.array([1,ImgLeft.shape[0],1])
    Pur = np.array([ImgLeft.shape[1],1,1])
    Pbr = np.array([ImgLeft.shape[1],ImgLeft.shape[0],1])

    lL = np.cross(Pul,Pbl)
    lR = np.cross(Pur,Pbr)

    plt.figure(1)
    plt.imshow(ImgRight)
    plt.axis('off')
    lLim,rLim = plt.ylim()
    for i in range(0,PtsLeft.shape[0]):
        e = np.matmul(F_matrix,np.transpose(np.hstack([PtsLeft[i,:],1])))
        PL = np.cross(e,lL)
        PR = np.cross(e,lR)
        x = np.array([PL[0]/PL[2],PR[0]/PR[2]])
        y = np.array([PL[1]/PL[2],PR[1]/PR[2]])
        plt.plot(x,y,c='b',linewidth=0.5)
        
    plt.scatter(PtsRight[:,0],PtsRight[:,1],c='r',marker='o',s=10)
    plt.ylim(lLim,rLim)

    plt.figure(2)
    plt.imshow(ImgLeft)
    plt.axis('off')
    lLim,rLim = plt.ylim()
    for i in range(0,PtsRight.shape[0]):
        e = np.matmul(np.transpose(F_matrix),np.transpose(np.hstack([PtsRight[i,:],1])))
        PL = np.cross(e,lL)
        PR = np.cross(e,lR)
        x = np.array([PL[0]/PL[2],PR[0]/PR[2]])
        y = np.array([PL[1]/PL[2],PR[1]/PR[2]])
        plt.plot(x,y,c='b',linewidth=0.5)
    
    plt.scatter(PtsLeft[:,0],PtsLeft[:,1],c='r',marker='o',s=10)
    plt.ylim(lLim,rLim)
    plt.show()

# This is a wrapper for opencv's ORB function. It removes duplicate
# points which would otherwise cause problems for RANSAC (because you would
# be likely to sample duplicate points and therefore your linear system
# would not have enough independent rows). This function also visualizes 
# corresponding points between two images. Corresponding points will be matched 
# by a line of random color. You do not need to modify anything in this 
# function, although you can if you want to.
def matchAndShowCorrespondence(imgA,imgB):
    
    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(imgA,None)
    kp2, des2 = orb.detectAndCompute(imgB,None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw all matches.
    img3 = cv2.drawMatches(imgA,kp1,imgB,kp2,matches,None,flags=2)

    # Extract matched keypoints 
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    matches_kp1 = np.asarray(list_kp1)
    matches_kp2 = np.asarray(list_kp2)

    # Remove duplicate matches 
    CombineReduce = np.unique(np.concatenate((matches_kp1,matches_kp2),axis=1),axis=0)
    matches_kp1 = CombineReduce[:,0:2]
    matches_kp2 = CombineReduce[:,-2:]

    # Display results
    fig = plt.figure()
    plt.imshow(img3)
    plt.axis('off')
    plt.show()

    print('Saving visualization to vis_arrows.jpg\n')
    fig.savefig(os.path.dirname(__file__) + '/vis_arrows.png',bbox_inches='tight',dpi=300)

    return matches_kp1, matches_kp2

# Display correspondences given a set of matches. You don't need to change this.
def showCorrespondence(imgA, imgB, matches_kp1, matches_kp2):
    
    imgA = img_as_float32(imgA)
    imgB = img_as_float32(imgB)

    fig = plt.figure()
    plt.axis('off')

    Height = max(imgA.shape[0],imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    numColors = imgA.shape[2]
    
    newImg = np.zeros([Height, Width, numColors])
    newImg[0:imgA.shape[0],0:imgA.shape[1],:] = imgA  
    newImg[0:imgB.shape[0],-imgB.shape[1]:,:] = imgB
    plt.imshow(newImg)
    
    shift = imgA.shape[1]
    for i in range(0,matches_kp1.shape[0]):
        
        r = lambda: random.randint(0,255)
        cur_color = ('#%02X%02X%02X' % (r(),r(),r()))
        
        x1 = matches_kp1[i,1]
        y1 = matches_kp1[i,0]
        x2 = matches_kp2[i,1]
        y2 = matches_kp2[i,0]

        x = np.array([x1, x2])
        y = np.array([y1, y2 + shift])
        plt.plot(y,x,c=cur_color,linewidth=0.5)

    plt.show()
