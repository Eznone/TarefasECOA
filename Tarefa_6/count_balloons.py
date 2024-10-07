import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from processes import *

# Obtaining desired comparee image
compareeImage = cv2.imread("../images/Tarefa_6/balloons.jpeg")

# Global variables --------------------------------
kernel = np.ones((5,5),np.uint8)

# Functions ---------------------------------------
def threshold_balloons(processed_image):
    _, newImage = cv2.threshold(processed_image, 50, 150, cv2.THRESH_BINARY_INV)
    newImage = cv2.dilate(newImage,kernel,iterations = 7)
    contours, _ = cv2.findContours(newImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = len(contours)

    return num_objects, newImage

def hough_circle(originalImage):

    image = originalImage.copy()
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=40, maxRadius = 100)
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

    return num_circles, originalImage


# Main Code ---------------------------------------

# Getting a shape for the balloons
processed_image = process_image(compareeImage)
#num_balloons, threshed_balloons = threshold_balloons(processed_image)
num_balloons, hough_balloons = hough_circle(processed_image)

cv2.imshow("Number of Balloons: %d" % num_balloons, hough_balloons)
cv2.waitKey(0)