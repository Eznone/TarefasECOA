import cv2
import argparse
import numpy as np





def gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    return gradient_magnitude

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get the image dimensions
height, width = image.shape

# Using 4 becuase it can make a matrix of 16 pieces 
piece_height = height // 4
piece_width = width // 4
counter = 0

# Making the 16 equal pieces 
for i in range(4):
    for j in range(4):
        # Following lines calculate beginning and end of each piece i want
        y_start = i * piece_height
        y_end = y_start + piece_height
        x_start = j * piece_width
        x_end = x_start + piece_width
        
        # Using image and its [] to make new pieces
        piece = image[y_start:y_end, x_start:x_end]
        
        # To show the image
        piece = cv2.GaussianBlur(piece, (11, 11), 0)
        piece = gradient_magnitude(piece)


        cv2.imshow(f'Piece {counter}', piece)
        cv2.waitKey(0)
        
        counter += 1

