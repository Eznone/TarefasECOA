import cv2
import numpy as np
import argparse


def concatinateImages(images):
    newImage = images[0]
    for image in images:
        image = cv2.resize(image, dsize = (0, 0), fx = 0.5, fy = 0.5)
        cv2.hconcat(newImage, image)
    return cv2.vconcat(newImage)

# Loading Image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
images = []
# Making kernel
kernel = np.ones((5,5),np.uint8)

# Get the image dimensions
height, width = image.shape

# Erosion method
erosion = cv2.erode(image, kernel, iterations = 1)
images.append(erosion)
cv2.imshow("Erosion", erosion)

# Dilation
dilation = cv2.dilate(image, kernel, iterations = 1)
images.append(dilation)


# Concatinating images
fullImage = concatinateImages(images)

cv2.imshow("Full Image", fullImage)
cv2.waitKey(0)