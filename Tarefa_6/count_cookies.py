import cv2
import numpy as np
import argparse
import math
from image_operations import *
from image_processes import *

# Global variables --------------------------------------------------
images = []
kernel = np.ones((5,5),np.uint8)
comparee_image = cv2.imread("../images/Tarefa_6/cookies.jpeg")

# Functions ---------------------------------------------------------
def threshold_cookies(processed_image):
    _, newImage = cv2.threshold(processed_image, 50, 200, cv2.THRESH_BINARY_INV)
    newImage = cv2.dilate(newImage, kernel, iterations = 10)
    #for i in range(0,5):
        #newImage = cv2.morphologyEx(newImage, cv2.MORPH_CLOSE, kernel)

    return newImage

def hough_circle(original_image):
    image = original_image.copy()
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=10, minRadius=40, maxRadius = 190)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Counting the circles
        num_circles = circles.shape[1]

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(original_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(original_image, center, radius, (255, 0, 255), 3)

    return num_circles, original_image

# Main ---------------------------------------------------------------
processed_image = process_image(comparee_image)
threshed_cookies = threshold_cookies(processed_image)
num_cookies, houghed_cookies = hough_circle(threshed_cookies)

cv2.imshow("Number of Cookies %d" % num_cookies, threshed_cookies)
cv2.waitKey(0)