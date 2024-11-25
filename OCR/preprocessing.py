import cv2
import numpy as np

def preprocess_image(image_path, target_height=48, target_width=400):
    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale (single channel)
    
    # Resize the image to the target dimensions (400x48)
    img_resized = cv2.resize(img, (target_width, target_height))
    
    # Normalize the image to [0, 1] range
    img_resized = img_resized / 255.0
    
    # Rotate the image 90 degrees clockwise
    img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
    
    # Add batch and channel dimensions (for grayscale: (1, 400, 48, 1))
    img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    
    return img_resized