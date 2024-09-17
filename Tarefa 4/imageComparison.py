import cv2
import numpy as np
import argparse
import math

# Global variables
images = []
kernel = np.ones((5,5),np.uint8)
coinComp = cv2.imread("../images/coin.png")

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

def gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    gradient_direction = np.arctan2(grad_y, grad_x)
    return (gradient_magnitude, gradient_direction)

def non_max_suppression(gradient_magnitude, gradient_direction):
    # Get the dimensions of the gradient magnitude image
    rows, cols = gradient_magnitude.shape
    
    # Create an empty array to store the suppressed image
    suppressed = np.zeros((rows, cols), dtype=np.uint8)
    
    # Convert gradient directions from radians to degrees
    angle = np.rad2deg(gradient_direction) % 180  # Limit the angle to [0, 180)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = 255
            r = 255
            
            # Determine the neighboring pixels to compare against
            # Angle 0 degrees (horizontal)
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            # Angle 45 degrees (diagonal)
            elif 22.5 <= angle[i,j] < 67.5:
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            # Angle 90 degrees (vertical)
            elif 67.5 <= angle[i,j] < 112.5:
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            # Angle 135 degrees (diagonal)
            elif 112.5 <= angle[i,j] < 157.5:
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            # Suppress non-maximum pixels
            if gradient_magnitude[i,j] >= q and gradient_magnitude[i,j] >= r:
                suppressed[i,j] = gradient_magnitude[i,j]
            else:
                suppressed[i,j] = 0
    return suppressed

def processImage(image):
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #newImage = cv2.GaussianBlur(newImage, (5, 5), 0)
    #_, newImage = cv2.threshold(newImage, 155, 255, cv2.THRESH_BINARY)
    #newImage = cv2.dilate(newImage, kernel, iterations = 1)

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
#images.append(originalImage)

# Processing image obtained
processedCoin = processImage(coinComp, )
#processedCoin = cv2.cvtColor(coinComp, cv2.COLOR_BGR2GRAY)
processedImage = processImage(originalImage)
images.append(processedImage)
images.append(processedCoin)

# Comparing images
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
for method in methods:
    tempImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (w, h, top_left, bottom_right, result) = imageComparitor(processedImage, processedCoin, method)
    threshold = 0.8
    loc = np.where(result >= threshold)
    #print(loc)
    #cv2.rectangle(tempImage, top_left, bottom_right, (0, 0, 0), 2)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(tempImage, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
    images.append(tempImage)

# Making grid layout
presentationImage = auto_image_grid(images)
cv2.imshow("Presentation Image", presentationImage)
cv2.waitKey(0)