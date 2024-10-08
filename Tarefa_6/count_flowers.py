import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from Tarefa_6.image_operations import *
from Tarefa_6.image_processes import *

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

def filter_labels(num_labels, labels, comparee_image):
    height, width, _ = comparee_image.shape
    #comparee_image_size = height * width
    print("Size of image: %d" % (height * width))

    for label in range(1, num_labels):
        # Count the number of pixels with the current label
        label_size = np.sum(labels == label)
        print("Label %d pixels" % label_size)
        # Check and Remove labels that are considered small
        if label_size < 30000:
            num_labels -= 1
            print("To be removed")
            label_to_remove = label
            mask = (labels == label_to_remove)
            comparee_image[mask] = 0

    return num_labels, comparee_image



# Main Code --------------------------------------------------
# Converting BGR image to HSV image
hsv_image = cv2.cvtColor(comparee_image, cv2.COLOR_BGR2HSV)

# Using Close method to get rid of noise
red_mask = make_red_mask()
red_clean_mask = clean_mask(red_mask)

# Apply the mask to the original image
image = comparee_image.copy()
red_flowers = cv2.bitwise_and(image, image, mask=red_clean_mask)

# Cleaning the red_flowers variable of noise and mistaken flowers
gray_red_flowers = cv2.cvtColor(red_flowers, cv2.COLOR_BGR2GRAY)
num_labels, labels = cv2.connectedComponents(gray_red_flowers)
num_labels, red_flowers = filter_labels(num_labels, labels, red_flowers)


# Display results
cv2.imshow("Number of Red Flowers %d" % (num_labels - 1), red_flowers)
cv2.waitKey(0)
cv2.destroyAllWindows()