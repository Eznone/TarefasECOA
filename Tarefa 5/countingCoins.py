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
  # Comparing images
  methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
  #methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED']
  for method in methods:
      # Preping temporaray image for boxes to be drawn on
      tempImageRGB = originalImage.copy()
      (w, h, top_left, bottom_right, result) = imageComparitor(processedImage, processedCoin, method)

      # Maxing a limit for some of the methods in the methods array
      threshold = 0.89
      loc = np.where(result >= threshold)
      
      # Un comment next line and comment the 2 lines after that to get only one result
      #cv2.rectangle(tempImage, top_left, bottom_right, (0, 0, 0), 2)
      for pt in zip(*loc[::-1]):
          cv2.rectangle(tempImageRGB, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

      images.append(tempImageRGB)

  # Making grid layout
  presentationImage = auto_image_grid(images)
  cv2.imshow("Presentation Image", presentationImage)
  cv2.waitKey(0)

def hough_circle(originalImage, coinImage):
   grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
   grayImage = cv2.medianBlur(grayImage, 5)
# Main ----------------------------------------------------------------------

# First method is matching templates as the function suggests
template_match(image.copy(), coinComp.copy())