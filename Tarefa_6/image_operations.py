import cv2
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt

def getTerminalImage():
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

    cv2.imshow("result", res)
    cv2.waitKey(0)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(paste, top_left, bottom_right, (0, 0, 0), 2)

    # Display the results
    return(w, h, top_left, bottom_right, res)

def display_image(image, title="Image"):
    """
    Displays an image using matplotlib in a Jupyter Notebook.

    :param image: The image to display (numpy array).
    :param title: Title for the displayed image.
    """
    # Convert BGR (OpenCV format) to RGB (matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def detect_circles(original_image):
    grayImage = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 5)

    rows = grayImage.shape[0]
    circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=40, maxRadius = 80)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Counting the circles
        num_circles = circles.shape[1]

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(original_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(original_image, center, radius, (255, 0, 255), 3)

    return num_circles, original_image
