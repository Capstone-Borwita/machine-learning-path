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

def rotate_if_necessary(image_path):
    """
    Resets an image that has been rotated 90°/180°/270°.
    If the image is in landscape but left has more white pixels, rotates 180°.
    
    Parameters:
        image_path : str
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # If the image is in landscape
    if image.shape[1] >= image.shape[0]:
        # Convert image to grayscale
        image_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and then Otsu's thresholding
        image_binary = cv2.GaussianBlur(image_binary, (5, 5), 0)
        _, image_binary = cv2.threshold(image_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Rotate by 180° if the left half has more white pixels than the right half
        if not compare_white_pixels(image_binary):
            image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        # If the image is in portrait, make it horizontal
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert image to grayscale
        image_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and then Otsu's thresholding
        image_binary = cv2.GaussianBlur(image_binary, (5, 5), 0)
        _, image_binary = cv2.threshold(image_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Rotate by 180° if the left half has more white pixels than the right half
        if not compare_white_pixels(image_binary):
            image = cv2.rotate(image, cv2.ROTATE_180)

    return image

# Path to the input image
image_path = "C:/Adhi/Belanar/Dicoding/scrapping/images/85e84f51-f4be-43a1-841d-bbda8323c4bc.jpeg"

# Process the image
document = rotate_if_necessary(image_path)

# Display the document
plt.axis("off")
plt.imshow(cv2.cvtColor(document, cv2.COLOR_BGR2RGB))
plt.show()
