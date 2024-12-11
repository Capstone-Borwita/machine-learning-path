# Segmentation and Cropping

This folder contains utilities for detecting, segmenting, and cropping objects in images, specifically focused on Indonesian ID cards (KTP). The tools utilize a YOLO-based segmentation model and offer functionalities for validation and geometric transformations.

---

## Features
1. **Object Detection and Segmentation:** Uses a pre-trained YOLO model for segmenting objects in input images.
2. **Validation of KTP Shape:** Ensures that the detected KTP has a valid rectangular shape.
3. **Validation of KTP Completeness:** Checks that the segmented KTP does not touch any image boundaries, ensuring it is entirely visible.
4. **Perspective Transformation:** Automatically crops and transforms the detected KTP for normalized output.
5. **Fallback Handling:** Employs bounding box-based cropping if the precise quadrilateral points are unavailable.

## How It Works

### Overview
The workflow involves:
1. Loading an input image.
2. Performing object detection and segmentation using YOLO.
3. Validating the detected KTP for completeness and rectangularity.
4. Applying geometric transformations (perspective warping) or fallback cropping using bounding boxes.
5. Outputting the processed and cropped KTP image.

### KTP as Whole Check
- The function ensures that the KTP mask does not touch any image boundaries by examining the top, bottom, left, and right edges of the mask.
- If any part of the mask touches these edges, the KTP is deemed incomplete, and an error message is returned.

### Cropping and Transformations
- **Four-Point Transformation:** If the KTP mask yields a valid quadrilateral, the script performs a perspective transformation to normalize the KTP appearance.
- **Bounding Box Cropping:** If the four points cannot be determined, a fallback method uses the bounding rectangle of the KTP mask for cropping.

### YOLO Model Integration
- The segmentation model (`segmentationCrop.pt`) is loaded using the `ultralytics` library.
- Predictions generate segmentation masks that are analyzed for contours and rectangularity.

## Usage

### Using the Python Script
The `segmentationCrop.py` script processes a single image:

#### Example:
```python
from segmentationCrop import process_single_ktp_image

input_image = "path/to/input_image.jpg"
result = process_single_ktp_image(input_image)

if isinstance(result, str):
    print(result)  # Error message or warning
else:
    cv2.imwrite("path/to/output_image.jpg", result)
```

## Functions and Modules

### segmentationCrop.py
#### `process_single_ktp_image(input_image_path)`
Processes a single image to detect and crop a KTP.

- **Input:** Path to an image file.
- **Output:**
  - Cropped KTP image (if successful).
  - Error message (if validation fails).

#### Sub Functions:
1. **`resizer(image, width=500):`**
   Resizes the image to a fixed width while maintaining aspect ratio.

2. **`is_valid_rectangle(points, tolerance=10):`**
   Validates that a set of four points forms a rectangle within a tolerance.

3. **`find_four_points(mask):`**
   Extracts four corner points of the largest contour matching a rectangle.

4. **`is_whole_ktp(mask):`**
   Ensures the segmented KTP does not touch any image boundary. If any edge of the mask touches the top, bottom, left, or right border, the KTP is flagged as incomplete.

5. **`wrap_ktp(mask, original_image):`**
   Performs perspective transformation using four points if available. If not, applies fallback cropping based on the bounding rectangle of the mask.

## Output
- Cropped image of the KTP.
- Error messages if processing fails due to:
  - KTP not being entirely visible.
  - Invalid rectangular shape.

## Notes
- Ensure the input images are well-lit and the KTP is entirely within the frame.
- Use fallback cropping for cases where perspective transformation is not feasible.

---

