import os
import cv2
from .utils.cropField import detect_and_crop
from .utils.OCR import extractText
from .utils.autoRotate import rotate_if_necessary
from .utils.segmentationCrop import process_single_ktp_image
from .utils.classification import classify_ktp


def ktp_ocr(
    input_image_path: str,
    segmentation_model_output_folder: str,
    rotate_model_output_folder: str,
    cropped_fields_folder: str,
):
    try:
        # Step 1: Classify KTP
        classification_result = classify_ktp(input_image_path)

        if not classification_result[0]:
            return "KTP tidak terdeteksi, harap coba lagi"

        # Step 2: Perform segmentation crop
        segmented_image = process_single_ktp_image(input_image_path)

        if type(segmented_image) is str:
            return segmented_image

        segmented_image_path = os.path.join(
            segmentation_model_output_folder, "segmented_image.jpg"
        )
        cv2.imwrite(segmented_image_path, segmented_image)

        # Step 3: Adjust image rotation
        rotated_image = rotate_if_necessary(segmented_image_path)

        rotated_image_path = os.path.join(
            rotate_model_output_folder, "rotated_image.jpg"
        )
        cv2.imwrite(rotated_image_path, rotated_image)

        # Step 4: Detect and crop fields
        cropped_images = detect_and_crop(
            rotated_image_path,
            cropped_fields_folder,
            confidence_threshold=0.5,
            save_crops=True,
        )

        # Step 5: Perform OCR on each cropped field
        if cropped_images:
            try:
                return extractText(cropped_images)
            except Exception as e:
                print(f"[ERROR] {e}")
        else:
            print("[INFO] No cropped fields to perform OCR on.\n")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
