import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define alphabets with uppercase English letters (A-Z), digits (0-9), and special characters
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*^_)(- .',"
max_str_len = 34  # max length of input labels
num_of_characters = len(alphabets) + 1  # +1 for CTC pseudo blank
num_of_timestamps = 100  # max length of predicted labels

# Function to convert label (string) to numerical representation

def label_to_num(label):
    label_num = []
    for ch in label:
        idx = alphabets.find(ch)
        if idx == -1:  # This means the character is not found in the alphabet
            raise ValueError(f"Character '{ch}' not in alphabet.")
        label_num.append(idx)
    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


# Function to load, resize, rotate and preprocess the image
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

def extractText(croppedImage):
    # Load the model
    modelNIK = tf.keras.models.load_model(r'models\ctc_crnn_nik.h5')
    model = tf.keras.models.load_model(r'models\ar_ver2.h5')
    extractedText = {}
    # Load and preprocess the image
    for cropped in croppedImage:
        image_path = croppedImage[cropped]  # Update with your image path
        image = preprocess_image(image_path)

        # Make the prediction
        preds = modelNIK.predict(image) if cropped == "NIK" else model.predict(image)

        # CTC Decode
        input_length = np.ones(preds.shape[0]) * preds.shape[1]  # Use the length of the output sequence
        decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)[0][0])

        # Convert the predicted labels back to text
        prediction = num_to_label(decoded[0])
        extractedText[cropped] = prediction

        # Plot the image and display the predicted label
        plt.imshow(cv2.rotate(image[0, :, :, 0], cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
        plt.title(f'Predicted Label: {prediction}')
        plt.show()

        # Print the prediction
    return extractedText