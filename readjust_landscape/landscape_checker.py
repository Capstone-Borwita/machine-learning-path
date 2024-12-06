import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_white_pixels(image):
    """
    Returns True if the right half of the image
    has more white pixels than the left half.
    
    Parameters:
        image : np.ndarray
    """
    width = image.shape[1]
    left_region = image[:, :int(width / 2)]
    right_region = image[:, int(width / 2):]

    left_white_pixels = np.sum(left_region == 255)
    right_white_pixels = np.sum(right_region == 255)
        
    return right_white_pixels > left_white_pixels

def check_and_display_message(image_path):
    """
    Checks if the image is in portrait orientation and if the right half has
    more white pixels than the left. Displays a message if the condition is met.
    
    Parameters:
        image_path : str
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # If the image is in portrait orientation
    if image.shape[1] < image.shape[0]:
        # Convert the image to grayscale
        image_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and then Otsu's thresholding
        image_binary = cv2.GaussianBlur(image_binary, (5, 5), 0)
        _, image_binary = cv2.threshold(image_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Check if the right half has more white pixels than the left half
        if compare_white_pixels(image_binary):
            print("Negative")
    else:
        print("Positive")
    
    return image

# Path to the input image
image_path = "C:/Adhi/Belanar/Dicoding/scrapping/images/0dfdc63f-ced6-4fd9-b518-910e186225b3.jpeg"

# Check the image and display the message if necessary
document = check_and_display_message(image_path)
