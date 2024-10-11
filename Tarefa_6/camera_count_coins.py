import cv2
import numpy as np
import argparse
import math
from image_operations import *
from image_processes import *
from count_coins import *


# Global variables ---------------------------------------------------------
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

# Functions ----------------------------------------------------------------



# Main ---------------------------------------------------------------------

while True:
    # Capturing image frame-by-frame
    ret, frame = capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detecting coins and balloons
    coin_image, threshold_image = detect_coins_hough_circles(frame)
    coin_image = cv2.resize(coin_image, (threshold_image.shape[1], threshold_image.shape[0]))
    
    #coin_image = cv2.cvtColor(coin_image, cv2.COLOR_GRAY2BGR)

    threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)

    combined_image = np.hstack((coin_image, threshold_image))

    if ret:
        cv2.imshow("Coins",  combined_image)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()