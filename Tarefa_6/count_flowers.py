import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from processes import *
from image_processes import *

# Obtaining desired comparee image ---------------------------
comparee_image = cv2.imread("../images/Tarefa_6/flowers.jpeg")

# Global variables -------------------------------------------
kernel = np.ones((5,5),np.uint8)

# Functions --------------------------------------------------
def make_red_mask():
    # Defining HSV color ranges for Red
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Creating masks with the given ranges made before
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combining the two masks made
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def clean_mask(original_mask):
    clean_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations = 3)
    
    return clean_mask


# Main Code --------------------------------------------------
# Converting BGR image to HSV image
hsv_image = cv2.cvtColor(comparee_image, cv2.COLOR_BGR2HSV)

# Using Close method to get rid of noise
red_mask = make_red_mask()
red_clean_mask = clean_mask(red_mask)

# Apply the mask to the original image
red_flowers = cv2.bitwise_and(comparee_image, comparee_image, mask=red_clean_mask)

gray_red_flowers = cv2.cvtColor(red_flowers, cv2.COLOR_BGR2GRAY)

num_labels, labels = cv2.connectedComponents(gray_red_flowers)


# Display results
cv2.imshow("Number of Red Flowers %d" % num_labels, red_flowers)
cv2.waitKey(0)
cv2.destroyAllWindows()