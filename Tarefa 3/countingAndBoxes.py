import cv2
import numpy as np
import argparse
import math

# functions ----------------------------------------------------------------
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


# Main code ----------------------------------------------------------------
images = []
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayed, (11, 11), 0)
images.append(grayed)
images.append(blurred)

edged = cv2.Canny(blurred, 30, 100)
images.append(edged)

boxImage = grayed.copy()

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    if area > 100:
        (x, y , w, h) = cv2.boundingRect(c)
        cv2.rectangle(boxImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(boxImage, c, -1, (255, 0, 0), 2)

images.append(boxImage)



finalImage = auto_image_grid(images)
cv2.imshow("Final Image", finalImage)
cv2.waitKey(0)