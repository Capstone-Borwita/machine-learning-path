import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224, 224)
CLASS_NAMES = ["KTP", "Not KTP"]

root = Path(__file__).parent.parent
model_path = root / "models/ktpClassifier.keras"
model = tf.keras.models.load_model(model_path)

def classify_ktp(image_path):
    if not os.path.exists(image_path):
        return f"File not found: {image_path}"

    image = cv2.imread(image_path)
    if image is None:
        return f"Failed to load image: {image_path}"

    # Normalize
    image_resized = cv2.resize(image, IMG_SIZE)
    image_normalized = image_resized / 255.0
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    # Prediction
    prediction = model.predict(input_data)[0][0]
    is_ktp = prediction < 0.5
    confidence = 1 - prediction if is_ktp else prediction

    return (image,confidence) if is_ktp else "Foto Tidak Memuat KTP, Coba Lagi"
