#!/usr/bin/env python

import sys
import cv2
import numpy as np


# build the mapping
def buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy):
    map_x = np.zeros((Hd,Wd), np.float32)
    map_y = np.zeros((Hd,Wd), np.float32)

    # Precompute sin and cos tables to speed up initialization
    # in the following row/col loop
    sincosTheta = []
    pi2 = 2.0 * np.pi
    for x in range(0, int(Wd - 1)):
        theta = (float(x) / float(Wd)) * pi2
        sincosTheta.append((np.sin(theta), np.cos(theta)))
    
    for y in range(0, int(Hd - 1)):
        r = (float(y) / float(Hd)) * (R2 - R1) + R1
        for x in range(0, int(Wd - 1)):
            sint, cost = sincosTheta[x]
            xS = Cx + r * sint
            yS = Cy + r * cost
            map_x.itemset((Hd - 1 - y, x), int(xS))
            map_y.itemset((Hd - 1 - y, Wd - 1 - x), int(yS))
        
    return map_x, map_y


# do the unwarping 
def unwarp(img, xmap, ymap):
    output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    #output = cv2.remap(img, xmap, ymap, cv2.INTER_CUBIC)
    return output


if __name__ == '__main__':

    # Center of the picture for the left eye in original image
    CxL = 525
    CyL = 540
    
    # Center of the picture for the right eye in original image
    CxR = 1390
    CyR = CyL # assume that they are on the same line

    # inner and outer donut radius (region of interest in the warped
    # image where we can actually see the environment)
    R1 = 150
    R2 = 355

    # Read original input image with two warped pictures in it
    if len(sys.argv) < 2:
        print('Please provide input image file name')
        sys.exit(1)

    img = cv2.imread(sys.argv[1])

    # Cut out only relevant parts for left and right eye.
    # These are two square images with 2*R2 size
    leftInput = img[CyL-R2 : CyL+R2, CxL-R2 : CxL+R2]
    rightInput = img[CyR-R2 : CyR+R2, CxR-R2 : CxR+R2]
    
    # Our source and destination image sizes
    Ws = leftInput.shape[0] # source width
    Hs = leftInput.shape[1] # source height
    Wd = 2.0* ((R2 + R1) / 2) * np.pi # dest width
    Hd = (R2 - R1) # dest height

    # Since left and right images are the same, we can use
    # the same centers coordinats for them
    Cx = int(Ws / 2)
    Cy = int(Hs / 2)
    
    # Build the pixel map for unwarping.
    # Since our left and right images have same dimensions
    # we can use the same map for both
    print("Initializing unwarping matricies...")
    xmap, ymap = buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy)
    print("Initialization completed. Unwarping...")

    # Unwarping
    resultL = unwarp(leftInput, xmap, ymap)
    resultR = unwarp(rightInput, xmap, ymap)
    # Scale down if desired
    #resultL = cv2.resize(resultL, (int(Wd / 2), int(Hd / 2)))
    #resultR = cv2.resize(resultR, (int(Wd / 2), int(Hd / 2)))

    # Displaying results
    cv2.namedWindow('Left', cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow('Left', resultL)
    
    cv2.namedWindow('Right', cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow('Right', resultR)

    # Save left and right image to file
    cv2.imwrite('data/left.png', resultL)
    cv2.imwrite('data/right.png', resultR)

    # Wait until key pressed and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
