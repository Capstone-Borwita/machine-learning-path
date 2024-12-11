
# KTP Classifier

This repository contains the code for training and testing a binary image classifier using Keras. The model is designed to distinguish between images of Kartu Tanda Penduduk (KTP) Indonesia and other that are not (non-KTP). This project is particularly developed with the purpose of being used as part of a process to KTP extraction.

---

## Table of Contents

- [Features](#features)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Usage](#Usage)

---

## Features

- **Data Augmentation**: Real-time transformations including rotation, zoom, and brightness adjustments to improve model robustness.
- **Transfer Learning**: Built on MobileNetV2 for efficient feature extraction and performance.
- **Fine-tuning**: Optimized by unfreezing specific layers for domain-specific learning.

---

## Setup and Installation

### Prerequisites

- Python 3.7+
- TensorFlow
- OpenCV
- Matplotlib
- Scikit-learn


## Training the Model

The training notebook `trainKTPClassifierKeras.ipynb` covers:

### 1. **Dataset Details**
- **Number of Images**: 435 KTP and 435 non-KTP images.
- **Preprocessing**: Images resized to 224x224 pixels.
- **Splitting**: 20% used for validation.

### 2. **Data Augmentation**

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    brightness_range=[0.8, 1.2],
    zoom_range=0.3,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)
```

### 3. **Model Architecture**

```python
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(1, activation="sigmoid")
])
```

### 4. **Callbacks**

```python
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath="model_best.keras", monitor="val_loss", save_best_only=True, verbose=1)
```

---

## Testing the Model

The testing notebook `testKerasModel.ipynb` demonstrates:

1. **Image Preprocessing**

```python
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, IMG_SIZE)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0).astype(np.float32), image
```

2. **Prediction and Confidence**

```python
def predict_image(image_path):
    input_data, original_image = preprocess_image(image_path)
    prediction = model.predict(input_data)[0][0]
    predicted_label = CLASS_NAMES[0] if prediction < 0.5 else CLASS_NAMES[1]
    return predicted_label, confidence, original_image
```

---


## Usage

To use the model for classification, follow these steps:

1. **Load the Trained Model**:
   Load the model weights from the saved file (`model.h5`).

2. **Run Inference**:
   Use the provided `predict` function to classify new images as `KTP` or `Non-KTP`.

### Example Code

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Load and preprocess image
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image / 255.0
image = np.expand_dims(image, axis=(0, -1))

# Predict
prediction = model.predict(image)
label = 'KTP' if prediction[0] > 0.5 else 'Non-KTP'
print(f'The image is classified as: {label}')
```

---

