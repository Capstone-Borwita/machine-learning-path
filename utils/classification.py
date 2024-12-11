from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from pathlib import Path


# Load the model
model = load_model(Path(__file__).parent.parent / "models/ktpClassifier.keras")


def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def classify_ktp(image_path):
    try:
        processed_image = preprocess_image(image_path, (224, 224))
    except:
        return "Gambar gagal diproses"

    prediction = model.predict(processed_image)[0][0]
    is_ktp = prediction < 0.5
    confidence_percentage = (1 - prediction) * 100

    return (is_ktp, confidence_percentage)
