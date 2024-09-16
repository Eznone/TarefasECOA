import cv2
import numpy as np
import argparse
import math

# Global variables
images = []
kernel = np.ones((5,5),np.uint8)
coinComp = cv2.imread("../images/coinComp2.png")

# Functions -----------------------------------------------------------------
def getImage():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help = "Path to image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    return image

def auto_image_grid(images, grid_size=None):
    """
    Arranges images in a grid layout based on the number of images.
    
    Args:
        images (list): List of images (must be of the same size or will be resized automatically).
        grid_size (tuple): (rows, cols) for grid layout. If None, the function calculates a square grid.
    
    Returns:
        grid_image: The resulting grid image.
    """

    # Check if there are any images
    if not images:
        raise ValueError("The image list is empty.")
    
    # Get the dimensions of the first image (assuming all images are the same size)
    image_height, image_width = images[0].shape[:2]
    new_height, new_width = image_height // 2, image_width // 2

    # Determine the grid size automatically if not provided
    num_images = len(images)
    if grid_size is None:
        grid_rows = math.ceil(math.sqrt(num_images))  # Number of rows
        grid_cols = math.ceil(num_images / grid_rows)  # Number of columns
    else:
        grid_rows, grid_cols = grid_size
    
    # Resize all images to the size of the first image (if they are not already the same size)
    resized_images = [cv2.resize(img, (image_width // 2, image_height // 2)) for img in images]
    
    # Add blank images to fill the grid if the number of images is not enough to fill it
    while len(resized_images) < grid_rows * grid_cols:
        blank_image = np.zeros_like(resized_images[0])  # Create a blank black image
        resized_images.append(blank_image)
    
    # Create the grid row by row
    grid_image = []
    for row in range(grid_rows):
        row_images = resized_images[row * grid_cols:(row + 1) * grid_cols]
        grid_image.append(cv2.hconcat(row_images))
    
    # Concatenate the rows vertically to get the full grid
    grid_image = cv2.vconcat(grid_image)
    
    return grid_image

def processImage(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    newImage = cv2.GaussianBlur(newImage, (11, 11), 0)
    newImage = cv2.dilate(newImage, kernel, iterations = 3)
    return newImage

def imageComparitor(image, template, meth):
    w, h  = template.shape[::-1]
    
    paste = image.copy()
    temp = image.copy()
    method = getattr(cv2, meth)

    # Using template match
    res = cv2.matchTemplate(temp, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(paste, top_left, bottom_right, (0, 0, 0), 2)

    # Display the results
    return(top_left, bottom_right, res)

# Main ----------------------------------------------------------------------

# Getting image through command line in console
originalImage = getImage()
#images.append(originalImage)

# Processing image obtained
processedCoin = processImage(coinComp)
processedImage = processImage(originalImage)
images.append(processedImage)
images.append(processedCoin)

# Comparing images
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
for method in methods:
    tempImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (top_left, bottom_right, result) = imageComparitor(processedImage, processedCoin, method)
    cv2.rectangle(tempImage, top_left, bottom_right, (0, 0, 0), 2)
    images.append(tempImage)

# Making grid layout
presentationImage = auto_image_grid(images)
cv2.imshow("Presentation Image", presentationImage)
cv2.waitKey(0)