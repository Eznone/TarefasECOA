import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from processes import *

# Obtaining desired comparee image
compareeImage = cv2.imread("../images/Tarefa 6/balloons.jpeg")


# Functions ---------------------------------------



# Main Code ---------------------------------------

processed = processImage(compareeImage)

cv2.imshow("threshed image", processed)
cv2.waitKey(0)

_, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow("threshed image", thresh)
cv2.waitKey(0)