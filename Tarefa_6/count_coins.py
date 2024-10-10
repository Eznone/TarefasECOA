import cv2
import numpy as np
import argparse
import math
from image_operations import *
from image_processes import *

# Global variables ----------------------------------------------------------
kernel = np.ones((5,5),np.uint8)
#coin_comp = cv2.imread("../images/coinComp4.png")

# Functions -----------------------------------------------------------------
# Convert the image to grayscale and apply thresholding
def detect_coins_water_shed(frame):
    gamma_image = adjust_gamma(frame, 1.0)

    gray = cv2.cvtColor(gamma_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    # To remove a lot of noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Determining object by seemingly expanding what an object is
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_CLOSE, kernel)

    # Canny detection to make edges for circle
    edges = cv2.Canny(sure_bg, 50, 150)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=50, minRadius=30, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Draw a rectangle for the center of the circle
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return frame, edges


    

def detect_coins(image):
    image = detect_coins_water_shed(image)
    
    return image
