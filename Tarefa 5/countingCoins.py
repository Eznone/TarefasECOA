import cv2
import numpy as np
import argparse
import math

from processes import *

# Global variables ----------------------------------------------------------
images = []
kernel = np.ones((5,5),np.uint8)
coinComp = cv2.imread("../images/coinComp4.png")
image = getImage()

# Functions -----------------------------------------------------------------
def template_match(originalImage, coinImage):
    processedCoin = processImage(coinImage, 0.25)
    processedImage = processImage(originalImage, 0.5)

    # Preping temporaray image for boxes to be drawn on
    tempImageRGB = originalImage.copy()
    (w, h, top_left, bottom_right, result) = imageComparitor(processedImage, processedCoin, 'TM_CCORR_NORMED')

    # Maxing a limit for some of the methods in the methods array
    threshold = 0.89
    loc = np.where(result >= threshold)

    
    # Counting the number of Objects
    num_objects_list = list(zip(*loc[::-1]))
    num_objects = len(num_objects_list)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(tempImageRGB, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    images.append(tempImageRGB)
    return num_objects

def hough_circle(originalImage):
  grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
  grayImage = cv2.medianBlur(grayImage, 5)

  rows = grayImage.shape[0]
  circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=40, maxRadius = 80)
  if circles is not None:
      circles = np.uint16(np.around(circles))

      # Counting the circles
      num_circles = circles.shape[1]

      for i in circles[0, :]:
          center = (i[0], i[1])
          # circle center
          cv2.circle(originalImage, center, 1, (0, 100, 100), 3)
          # circle outline
          radius = i[2]
          cv2.circle(originalImage, center, radius, (255, 0, 255), 3)
  images.append(originalImage)
  return num_circles

def watershed(originalImage):
    gammaImage = adjust_gamma(image, 1.0)
    grayImage = cv2.cvtColor(gammaImage, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    _, thresh = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY_INV)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(originalImage,markers)
    originalImage[markers == -1] = [255,0,0]

    # Count the number of unique markers excluding the background (label 1)
    unique_labels = np.unique(markers)
    num_objects = len(unique_labels) - 2 


    #cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    #cv2.imshow("dist_transform", dist_transform)
    #cv2.waitKey(0)

    images.append(originalImage)
    return num_objects

# Main ----------------------------------------------------------------------

# Template Matching to get # of Circles
num_template = template_match(image.copy(), coinComp.copy())
print("Number of Circles in Template Matching:%d" % num_template)

# Hough Circle Detection to get # of Circles
num_hough = hough_circle(image.copy())
print("Number of Circles in Hough:%d," % num_hough)

# Watershed Segmentation to get # of Circles
num_water = watershed(image.copy())
print("Number of Circles in WaterShed:%d" % num_water)


# Making grid layout
presentationImage = auto_image_grid(images)
cv2.imshow("Hough Image", presentationImage)
cv2.waitKey(0)