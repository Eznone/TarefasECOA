import cv2
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from processes import *

# Obtaining desired comparee image
compareeImage = cv2.imread("../images/Tarefa_6/balloons.jpeg")

# Functions ---------------------------------------


# Main Code ---------------------------------------
num_balloons, detected_image = detect_blobs(compareeImage)

cv2.imshow("%d amount of balloons" % num_balloons, detected_image)
cv2.waitKey(0)