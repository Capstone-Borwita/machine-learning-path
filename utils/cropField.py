from ultralytics import YOLO
import cv2
from pathlib import Path

# Initialize YOLO model
model = YOLO(Path(__file__).parent.parent / "models/Model_Detection_Label.pt")


# Detect objects and crop regions
def detect_and_crop(img, confidence_threshold=0.5):
    # Run YOLO inference
    results = model(img)
    boxes = results[0].boxes
    class_names = results[0].names

    cropped_images = {}
    for i, box in enumerate(boxes):
        confidence = box.conf.item()
        if confidence < confidence_threshold:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = img[y1:y2, x1:x2]
        class_id = int(box.cls.item())
        class_name = class_names[class_id]

        cropped_images[class_name] = cropped

    if len(cropped_images) != 3:
        classes = " dan".join(
            ", ".join(
                cropped_class
                for cropped_class in ["NIK", "Nama", "Alamat"]
                if cropped_class not in cropped_images
            ).rsplit(",", 1)
        )

        return (None, f"{classes} tidak dapat dibaca")

    return (cropped_images, None)
