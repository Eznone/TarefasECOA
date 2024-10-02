import cv2
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt

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
    kernel = np.ones((5,5),np.uint8)
    newImage = adjust_gamma(image, gamma)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    newImage = cv2.GaussianBlur(newImage, (5, 5), 0)
    newImage = cv2.dilate(newImage, kernel, iterations = 1)
    cv2.imshow("Processing", newImage)
    cv2.waitKey(0)
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

def create_border(originalImage, templateImage):

    original_height, original_width = originalImage.shape[:2]
    template_height, template_width = templateImage.shape[:2]

    vertical_border = max(0, (template_height) // 2)
    horizontal_border = max(0, (template_width - 1) // 2)

    bordered_image = cv2.copyMakeBorder(
        originalImage,
        vertical_border,
        vertical_border,
        horizontal_border,
        horizontal_border,
        borderType=cv2.BORDER_CONSTANT,
        # Remember this is in BGR format for blak
        value=[0, 0, 0]  
    )
    return bordered_image

def remove_border(bordered_image, originalImage, templateImage):

    original_height, original_width = originalImage.shape[::-1]
    template_height, template_width = templateImage.shape[::-1]

    vertical_border = max(0, (template_height) // 2)
    print(vertical_border)
    horizontal_border = max(0, (template_width) // 2)
    print(horizontal_border)

    cropped_image = bordered_image[
        vertical_border:bordered_image.shape[0] - vertical_border,
        horizontal_border:bordered_image.shape[1] - horizontal_border
    ]
    
    return cropped_image

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