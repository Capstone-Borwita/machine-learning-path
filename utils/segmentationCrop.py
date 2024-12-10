from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from imutils.perspective import four_point_transform

model = YOLO(Path(__file__).parent.parent / "models/segmentationCrop.pt")


def process_single_ktp_image(input_image_path):
    def resizer(image, width=500):
        height = int((image.shape[0] / image.shape[1]) * width)
        return cv2.resize(image, (width, height)), (width, height)

    # Check if KTP is a rectangle
    def is_valid_rectangle(points, tolerance=10):
        if points is None or len(points) != 4:
            return False
        distances = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]
        diagonals = [np.linalg.norm(points[0] - points[2]), np.linalg.norm(points[1] - points[3])]
        return (
            abs(distances[0] - distances[2]) < tolerance
            and abs(distances[1] - distances[3]) < tolerance
            and abs(diagonals[0] - diagonals[1]) < tolerance
        )

    # Check if masked KTP has 4 points
    def find_four_points(mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and is_valid_rectangle(approx.squeeze()):
                return approx.squeeze()
        return None

    # Check if KTP is whole
    def is_whole_ktp(mask):
        # Threshold the mask
        mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY) if len(mask_binary.shape) == 3 else mask_binary

        # Check if any pixel of the thresholded mask touch any border
        top_row = mask_binary[0, :]
        bottom_row = mask_binary[-1, :]
        left_col = mask_binary[:, 0]
        right_col = mask_binary[:, -1]

        # if somehow any of those column or row contain any white pixel (from the threshold), then it means the mask touched the border
        return not any([np.any(top_row),np.any(bottom_row),np.any(left_col),np.any(right_col)])

    # Wrap or Crop the KTP based on it's four points or bounding box
    def wrap_ktp(mask, original_image):
        mask_resized, size = resizer(mask)
        original_resized, _ = resizer(original_image)

        if len(mask_resized.shape) == 2:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        four_points = find_four_points(mask_resized)

        if four_points is None:
            # Fall back to bounding rectangle if no rectangle is detected
            mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
            mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY) if len(mask_binary.shape) == 3 else mask_binary
            x, y, w, h = cv2.boundingRect(mask_binary)
            scale_x, scale_y = original_image.shape[1] / size[0], original_image.shape[0] / size[1]
            return original_image[int(y * scale_y):int((y + h) * scale_y), int(x * scale_x):int((x + w) * scale_x)]

        # Scale four points to the original image size
        scale_x, scale_y = original_image.shape[1] / size[0], original_image.shape[0] / size[1]
        four_points_scaled = (four_points * [scale_x, scale_y]).astype(int)

        try:
            return four_point_transform(original_image, four_points_scaled)
        except Exception as e:
            print(f"Perspective transform error: {e}")
            return original_image

    # Load and process the image
    image = cv2.imread(input_image_path)
    results = model.predict(source=input_image_path, save=False, conf=0.5)

    for result in results:
        if result.masks is not None and len(result.masks.data) > 0:
            mask = (result.masks.data[0].cpu().numpy() > 0).astype(np.uint8) * 255

            # Check if the KTP is whole
            if not is_whole_ktp(mask):
                print("KTP is not Whole")
                return "KTP terdeteksi tidak lengkap. Pastikan seluruh bagian KTP terlihat dalam gambar."

            return wrap_ktp(mask, image)

