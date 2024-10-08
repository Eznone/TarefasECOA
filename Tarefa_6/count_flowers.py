import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from processes import *

# Obtaining desired comparee image ---------------------------
comparee_image = cv2.imread("../images/Tarefa_6/flowers.jpeg")

# Global variables -------------------------------------------
kernel = np.ones((5,5),np.uint8)

# Functions --------------------------------------------------



# Main Code --------------------------------------------------
hsv_image = cv2.cvtColor(comparee_image, cv2.COLOR_BGR2HSV)

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

# Using Close method to get rid of noise
kernel = np.ones((5, 5), np.uint8)
red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to the original image
red_flowers = cv2.bitwise_and(comparee_image, comparee_image, mask=red_mask_clean)

# Display results
cv2.imshow("Red Flowers", red_flowers)
cv2.waitKey(0)
cv2.destroyAllWindows()