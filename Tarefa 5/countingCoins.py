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
  
  for pt in zip(*loc[::-1]):
      cv2.rectangle(tempImageRGB, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

  images.append(tempImageRGB)

def hough_circle(originalImage):
  grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
  grayImage = cv2.medianBlur(grayImage, 5)

  rows = grayImage.shape[0]
  circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=40, maxRadius = 80)
  if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
          center = (i[0], i[1])
          # circle center
          cv2.circle(originalImage, center, 1, (0, 100, 100), 3)
          # circle outline
          radius = i[2]
          cv2.circle(originalImage, center, radius, (255, 0, 255), 3)
  images.append(originalImage)


# Main ----------------------------------------------------------------------

# First method is matching templates as the function suggests
template_match(image.copy(), coinComp.copy())
hough_circle(image.copy())
# Making grid layout
presentationImage = auto_image_grid(images)
cv2.imshow("Hough Image", presentationImage)
cv2.waitKey(0)