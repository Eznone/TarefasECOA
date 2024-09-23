import cv2
import numpy as np
import argparse
import math

# Global variables ----------------------------------------------------------
images = []
kernel = np.ones((5,5),np.uint8)
coinComp = cv2.imread("../images/coinComp4.png")

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
        images (list): List of images (can be a mix of grayscale and RGB).
        grid_size (tuple): (rows, cols) for grid layout. If None, the function calculates a square grid.
    
    Returns:
        grid_image: The resulting grid image.
    """

    # Check if there are any images
    if not images:
        raise ValueError("The image list is empty.")
    
    # Get the dimensions of the first image (assuming all images are the same size)
    image_height, image_width = images[0].shape[:2]
    
    # Determine the grid size automatically if not provided
    num_images = len(images)
    if grid_size is None:
        grid_rows = math.ceil(math.sqrt(num_images))  # Number of rows
        grid_cols = math.ceil(num_images / grid_rows)  # Number of columns
    else:
        grid_rows, grid_cols = grid_size
    
    # Resize and convert images as necessary
    processed_images = []
    for img in images:
        # Convert grayscale images to RGB for consistent display if necessary
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:  # Grayscale image with one channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize the image to the size of the first image
        resized_img = cv2.resize(img, (image_width, image_height))
        processed_images.append(resized_img)
    
    # Add blank images to fill the grid if the number of images is not enough to fill it
    while len(processed_images) < grid_rows * grid_cols:
        blank_image = np.zeros_like(processed_images[0])  # Create a blank black image
        processed_images.append(blank_image)
    
    # Create the grid row by row
    grid_image = []
    for row in range(grid_rows):
        row_images = processed_images[row * grid_cols:(row + 1) * grid_cols]
        grid_image.append(cv2.hconcat(row_images))
    
    # Concatenate the rows vertically to get the full grid
    grid_image = cv2.vconcat(grid_image)
    
    return grid_image

def adjust_gamma(image, gamma = 1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def processImage(image, gamma = 1.0):
    images.append(image)
    newImage = adjust_gamma(image, gamma)
    images.append(newImage)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    images.append(newImage)
    newImage = cv2.GaussianBlur(newImage, (5, 5), 0)
    images.append(newImage)
    newImage = cv2.dilate(newImage, kernel, iterations = 1)
    images.append(newImage)

    return newImage

def imageComparitor(image, template, meth):
    #cv2.imshow("template", template)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
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
    return(w, h, top_left, bottom_right, res)

# Main ----------------------------------------------------------------------

# Getting image through command line in console
originalImage = getImage()

# Processing image obtained
processedCoin = processImage(coinComp, 0.25)
processedImage = processImage(originalImage, 0.5)

# Comparing images
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
#methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED']
for method in methods:
    # Preping temporaray image for boxes to be drawn on
    tempImageRGB = originalImage.copy()
    (w, h, top_left, bottom_right, result) = imageComparitor(processedImage, processedCoin, method)

    # Maxing a limit for some of the methods in the methods array
    threshold = 0.89
    loc = np.where(result >= threshold)
    
    # Un comment next line and comment the 2 lines after that to get only one result
    #cv2.rectangle(tempImage, top_left, bottom_right, (0, 0, 0), 2)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(tempImageRGB, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    images.append(tempImageRGB)

# Making grid layout
presentationImage = auto_image_grid(images)
cv2.imshow("Presentation Image", presentationImage)
cv2.waitKey(0)