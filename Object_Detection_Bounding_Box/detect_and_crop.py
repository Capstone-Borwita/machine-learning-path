from ultralytics import YOLO
import cv2
import os

# Initialize YOLO model
model = YOLO("D:/Bangkit Capstone/Object-Detection-Bounding-Box/Object_Detection_Bounding_Box/Train result/train/weights/Model_Detection_Label.pt")  # Ensure relative path

# Detect objects and crop regions
def detect_and_crop(image_path, output_folder, confidence_threshold=0.5, save_crops=False):
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return []

    # Run YOLO inference
    results = model(img)
    boxes = results[0].boxes  

    cropped_images = []
    for i, box in enumerate(boxes):
        confidence = box.conf.item()
        if confidence < confidence_threshold:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = img[y1:y2, x1:x2]
        cropped_images.append(cropped)

        # Save cropped regions (optional)
        if save_crops:
            output_path = f"{output_folder}/crop_{i}.jpg"
            cv2.imwrite(output_path, cropped)
            print(f"Saved: {output_path}")
    return cropped_images

# Example usage
if __name__ == "__main__":
    input_image = "Object_Detection_Bounding_Box/inputs-ktp/test-ktp.jpg"  # Example input image
    output_folder = "output-crop"          # Output folder for cropped images
    detect_and_crop(input_image, output_folder, confidence_threshold=0.7, save_crops=True)
