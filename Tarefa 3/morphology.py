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
    resized_images = [cv2.resize(img, (image_width // 3, image_height // 3)) for img in images]
    
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

def apply_threshold(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def hysteresis_tracking(double_threshold_image):
    """
    Perform edge tracking by hysteresis on the double-thresholded image.
    
    Parameters:
        double_threshold_image (numpy.ndarray): The input image with weak and strong edges marked after double thresholding.

    Returns:
        numpy.ndarray: The final edge map after hysteresis tracking.
    """
    rows, cols = double_threshold_image.shape
    output = np.zeros((rows, cols), dtype=np.uint8)  # Create a new image for the output

    # Strong edges: pixels with value 255
    strong_edges = double_threshold_image == 255
    
    # Weak edges: pixels with value 100
    weak_edges = double_threshold_image == 100

    # Copy strong edges to the output directly
    output[strong_edges] = 255

    # Check weak edges and promote them only if connected to strong edges
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if weak_edges[i, j]:
                # Check if any of the 8-connected neighbors is a strong edge
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    output[i, j] = 255  # Promote weak edge to strong edge
                else:
                    output[i, j] = 0  # Discard weak edge
    
    return output

def canny_image(grayed):
    # This is my own canny function
    
    # Parameters: A grayed image

    # Returns: A canny processed image

    kernel = np.ones((5,5), np.uint8)
    images.append(grayed)
    # Making image Greyed
    blurred = cv2.GaussianBlur(grayed, (11, 11), 0)
    images.append(blurred)

    # Making gradient image
    magnitude, direction = gradient_magnitude(blurred)

    # Suppreseding gradient image
    non_max_suppressed = non_max_suppression(magnitude, direction)
    images.append(non_max_suppressed)
    
    # Thresholding the suppresed image
    double_thresholded = apply_threshold(non_max_suppressed)
    images.append(double_thresholded)

    # Making better edges through hysteresis tracking
    opening = cv2.dilate(double_thresholded, kernel, iterations = 1)
    #images.append(tracked)
    return opening

def convexHull(image, original, imageList):

    # Function uses convexHull contours to draw minimum points around object

    edge = canny_image(image)

    # Find contours in the image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour
    for contour in contours:
        # Find the convex hull of the contour
        hull = cv2.convexHull(contour)
        
        # Draw the convex hull on the image
        cv2.drawContours(original, [hull], -1, (255, 0, 0), 2)
    
    # cv2.imshow("Hull after", hull_image)

    return original


# Main code ---------------------------------------------------------------------

# Loading Image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

original = cv2.imread(args["image"])
image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
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
hull_image = convexHull(image, image,  images)
images.append(hull_image)

# Concatinating images
fullImage = auto_image_grid(images)

cv2.imshow("Full Image", fullImage)
cv2.waitKey(0)