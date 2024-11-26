import cv2
import numpy as np
from ultralytics import YOLO
from imutils.perspective import four_point_transform


def process_single_ktp_image(input_image_path):

    model = YOLO(r"models\segmentationCrop.pt")

    def resizer(image, width=500):
        h, w = image.shape[:2]
        height = int((h / w) * width)
        size = (width, height)
        return cv2.resize(image, size), size

    def is_rectangle(points):
        if points is None or len(points) != 4:
            return False
        dists = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]
        diagonal1 = np.linalg.norm(points[0] - points[2])
        diagonal2 = np.linalg.norm(points[1] - points[3])
        return (abs(dists[0] - dists[2]) < 10 and
                abs(dists[1] - dists[3]) < 10 and
                abs(diagonal1 - diagonal2) < 10)

    def wrap_ktp(mask, original_image):
        mask_resized, size = resizer(mask)
        original_resized = resizer(original_image)[0]
        if len(mask_resized.shape) == 2:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        four_points = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                four_points = np.squeeze(approx)
                if is_rectangle(four_points):
                    break

        if four_points is None or not is_rectangle(four_points):
            mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
            mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY) if len(mask_binary.shape) == 3 else mask_binary
            x, y, w, h = cv2.boundingRect(mask_binary)
            scale_x = original_image.shape[1] / size[0]
            scale_y = original_image.shape[0] / size[1]
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            return original_image[y:y + h, x:x + w]

        scale_x = original_image.shape[1] / size[0]
        scale_y = original_image.shape[0] / size[1]
        four_points_scaled = (four_points * [scale_x, scale_y]).astype(int)

        try:
            return four_point_transform(original_image, four_points_scaled)
        except Exception as e:
            print(f"Perspective transform error: {e}")
            return original_image

    # Load and process the image
    image_colored = cv2.imread(input_image_path)
    results = model.predict(source=input_image_path, save=False, conf=0.5)

    for result in results:
        if result.masks is not None and len(result.masks.data) > 0:
            mask = (result.masks.data[0].cpu().numpy() > 0).astype(np.uint8) * 255
            return wrap_ktp(mask, image_colored)

    print(f"No KTP detected in the image: {input_image_path}")
    return image_colored