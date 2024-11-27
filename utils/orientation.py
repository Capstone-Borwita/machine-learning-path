# Import necessary packages
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

# Load custom YOLOv8 model trained on KTP fields
model = YOLO(Path(__file__).parent.parent / "models/orientation.pt")

# Define the class ID you want to detect, e.g., ID field or name field in the KTP
TARGET_CLASS_ID = 0  # Adjust as necessary


def detect_field_in_rotated_image(
    image_path, target_class_id=TARGET_CLASS_ID, confidence_threshold=0.6
):
    """
    Rotates the image in increments and performs object detection until a target object is found.

    Args:
        image_path (str): Path to the image file.
        target_class_id (int): The class ID for the target object to detect.
        confidence_threshold (float): Confidence threshold for detections.

    Returns:
        None
    """
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Image not found or unable to load.")

    height, width = original_image.shape[:2]
    angle = 0
    best_confidence = 0
    best_detection = None
    best_rotated_image = None

    # Rotate image incrementally and detect
    while angle < 360:
        # Rotate the image by the current angle
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))

        # Perform inference on the rotated image
        results = model.predict(
            source=rotated_image, conf=confidence_threshold, save=False
        )

        # Check if target class is detected and has highest confidence so far
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            # Verify if the detected class matches the target class
            if class_id == target_class_id and confidence > best_confidence:
                # Update best detection information
                best_confidence = confidence
                best_detection = box
                best_rotated_image = rotated_image.copy()

        # Increase the angle for the next rotation
        angle += 10  # Adjust the rotation increment if needed

    # If a best detection was found, display it
    if best_detection is not None:
        x1, y1, x2, y2 = map(int, best_detection.xyxy[0].tolist())
        cv2.rectangle(
            best_rotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
        )

        # Prepare label with class name and confidence
        label = f"{model.names[target_class_id]}: {best_confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_offset_x, text_offset_y = x1, y1 - 5

        # Draw background rectangle for text label
        cv2.rectangle(
            best_rotated_image,
            (text_offset_x, text_offset_y - text_height - 2),
            (text_offset_x + text_width, text_offset_y),
            (0, 255, 0),
            -1,
        )

        # Place label text
        cv2.putText(
            best_rotated_image,
            label,
            (text_offset_x, text_offset_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Display the detected image with the highest confidence
        # plt.imshow(cv2.cvtColor(best_rotated_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"Detected {model.names[target_class_id]} with Highest Confidence")
        # plt.axis("off")  # Hide axis
        # plt.show()
        return best_rotated_image
    else:
        print("Target field not detected in any rotation.")
