import cv2
import numpy as np
import argparse
import math

# Functions ---------------------------------------------------------------------
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
    
    # Determine the grid size automatically if not provided
    num_images = len(images)
    if grid_size is None:
        grid_rows = math.ceil(math.sqrt(num_images))  # Number of rows
        grid_cols = math.ceil(num_images / grid_rows)  # Number of columns
    else:
        grid_rows, grid_cols = grid_size
    
    # Resize all images to the size of the first image (if they are not already the same size)
    resized_images = [cv2.resize(img, (image_width, image_height)) for img in images]
    
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

def convexHull(image):

    # Function uses convexHull contours to draw minimum points around object

    edge = cv2.Canny(image, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("thresh", edge)

    # Find contours in the image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the convex hull
    hull_image = np.zeros_like(image)
    cv2.imshow("Hull before", hull_image)

    # Iterate through each contour
    for contour in contours:
        # Find the convex hull of the contour
        hull = cv2.convexHull(contour)
        
        # Draw the convex hull on the image
        cv2.drawContours(hull_image, [hull], -1, (255, 0, 0), 2)
    
    cv2.imshow("Hull after", hull_image)

    return hull_image


# Main code ---------------------------------------------------------------------

# Loading Image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
images = []
# Making kernel
kernel = np.ones((5,5),np.uint8)

# Erosion method
erosion = cv2.erode(image, kernel, iterations = 1)
images.append(erosion)

# Dilation
dilation = cv2.dilate(image, kernel, iterations = 1)
images.append(dilation)

# Opening
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
images.append(dilation)

# Closing
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
images.append(closing)

# Convex Hull
hull_image = convexHull(image)
images.append(hull_image)

# Concatinating images
fullImage = auto_image_grid(images)

cv2.imshow("Full Image", fullImage)
cv2.waitKey(0)