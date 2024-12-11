from pathlib import Path
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import regex as re

# Define alphabets with uppercase English letters (A-Z), digits (0-9), and special characters
alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*^_)(- .',"
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
            ret += alphabets[ch]
    return ret


# Function to load, resize, rotate and preprocess the image
def preprocess_image(img, target_height=48, target_width=400):
    # Convert image to grayscale (single channel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


# Load the model
root = Path(__file__).parent.parent
modelNIK = tf.keras.models.load_model(root / "models/ctc_crnn_nik.h5")
model = tf.keras.models.load_model(root / "models/ar_ver4.h5")


def is_valid_nik(nik):
    nik_pattern = r"^\d{6}(?:(?:0[1-9]|[1-2][0-9]|3[0-1])|(?:[4-6][0-9]|7[0-1]))(?:0[1-9]|1[0-2])\d{5}[1-9]$"
    return bool(re.match(nik_pattern, nik))


def extractText(croppedImage):
    extractedText = {}
    # Load and preprocess the image
    for cropped in croppedImage:
        image_path = croppedImage[cropped]  # Update with your image path
        image = preprocess_image(image_path)

        # Make the prediction
        preds = modelNIK.predict(image) if cropped == "NIK" else model.predict(image)

        # CTC Decode
        input_length = (
            np.ones(preds.shape[0]) * preds.shape[1]
        )  # Use the length of the output sequence
        decoded = tf.keras.backend.get_value(
            tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)[
                0
            ][0]
        )

        # Convert the predicted labels back to text
        prediction = num_to_label(decoded[0]).strip()

        # Add extracted text into a dictionary
        extractedText[cropped] = prediction

        # Plot the image and display the predicted label
        plt.imshow(
            cv2.rotate(image[0, :, :, 0], cv2.ROTATE_90_COUNTERCLOCKWISE), cmap="gray"
        )
        plt.title(f"Predicted Label: {prediction}")
        plt.show()

    emptyTexts = [key for key, value in extractedText.items() if value == ""]
    if len(emptyTexts) > 0:
        classes = " dan".join(", ".join(emptyTexts).rsplit(",", 1))

        return f"{classes} tidak terbaca"

    if len(extractedText["NIK"]) < 16:
        return "NIK terpotong"
    elif not is_valid_nik(extractedText["NIK"]):
        return f"{extractedText['NIK']} tidak sesuai dengan format NIK"

    return extractedText
