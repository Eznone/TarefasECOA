import cv2
import numpy as np
import argparse

def gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    gradient_direction = np.arctan2(grad_y, grad_x)
    return (gradient_magnitude, gradient_direction)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
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
    # Here i use Canny instead of snakelets (Snaklets does better job but is far more costly)
    return thresh

def interpolate_points(edges, gradient_magnitude):
    # Find contours to determine points of interest
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Creating empty map of same size to make interpolation on top of
    interpolated = np.zeros_like(gradient_magnitude)
    
    # Iterate over contours to fill the interpolation map
    for c in contours:
        for point in c:
            x, y = point[0]
            interpolated[y, x] = gradient_magnitude[y, x]
    
    # Apply interpolation (can use smoothing techniques like GaussianBlur to approximate interpolation)
    interpolated = cv2.GaussianBlur(interpolated, (11, 11), 0)
    interpolated = cv2.normalize(interpolated, None, 0, 255, cv2.NORM_MINMAX)
    equalized = cv2.equalizeHist(interpolated.astype(np.uint8))
    
    return equalized

# def fill_interpolation(interpolated_image, original_image):
#     # Fill the image based on interpolated points, making sure the border has higher values to avoid confusion
#     filled_image = interpolated_image.copy()
#     filled_image[filled_image == 0] = original_image[filled_image == 0]
    
#     return filled_image

# def validate_segmentation(segmented_image, gradient_magnitude):
#     # Label connected components in segmented image
#     num_labels, labels_im = cv2.connectedComponents(segmented_image)
    
#     validated_image = np.zeros_like(segmented_image)
    
#     # Iterate over each labeled component
#     for label in range(1, num_labels):  # Skip background label
#         mask = np.uint8(labels_im == label)
#         edge_strength = cv2.mean(gradient_magnitude, mask=mask)[0]
        
#         # Validate: only keep components with strong enough edge strength
#         if edge_strength > 50:  # Threshold can be adjusted based on the image
#             validated_image[mask > 0] = 255
    
#     return validated_image





# Main code ---------------------------------------------------------------------

# Loading Image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

# Smoothing the image to get a better contour
image = cv2.imread(args["image"])
cv2.imshow("Original Image", image)

# Convert to grayscale and apply Gaussian Blur
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
original = image.copy()
image = cv2.GaussianBlur(image, (11, 11), 0)

cv2.imshow("Blurred Image", image)

# Finding Gradient Magnitude
(grad_mag, grad_dir) = gradient_magnitude(image)
cv2.imshow("Gradient Magnitude", grad_mag)

# Suppresing the Gradient
suppresed = non_maximum_suppression(grad_mag, grad_dir)
cv2.imshow("Suppresed Gradient", suppresed)

# Applying Thresholding onto the Gradient Magnitude
edges = apply_threshold(suppresed)
cv2.imshow("Thresholding", edges)

# Interpolating image using the correct gradient magnitude

interpolated2 = interpolate_points(edges, grad_mag)

cv2.imshow("Interpolated2", interpolated2)


# Filling the interpolation with original image data
# filled_image = fill_interpolation(interpolated, original)
# cv2.imshow("Filled Interpolation", filled_image)

# Segmenting the image
# _, segmented = cv2.threshold(interpolated, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("Segmented", segmented)

# Validating the segmentation using gradient magnitude
# validated = validate_segmentation(segmented, grad_mag)
# cv2.imshow("Validated", validated)
cv2.waitKey(0)