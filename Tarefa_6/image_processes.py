import cv2
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt

def adjust_gamma(image, gamma = 1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]


    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)


    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Bounding box measurements
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def process_image(image, gamma = 1.0):
    kernel = np.ones((5,5),np.uint8)
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    newImage = cv2.GaussianBlur(newImage, (5, 5), 0)
    # newImage = cv2.dilate(newImage, kernel, iterations = 1)
    return newImage

def clean_mask(original_mask):
    kernel = np.ones((5,5),np.uint8)
    clean_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations = 3)
    
    return clean_mask
