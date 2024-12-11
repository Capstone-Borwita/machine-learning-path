from .utils.cropField import detect_and_crop
from .utils.OCR import extractText
from .utils.autoRotate import rotate_if_necessary
from .utils.segmentationCrop import process_single_ktp_image
from .utils.classification import classify_ktp


def ktp_ocr(input_image_path: str):
    try:
        # Step 1: Classify KTP
        classification_result = classify_ktp(input_image_path)

        if not classification_result[0]:
            return "KTP tidak terdeteksi, harap coba lagi"

        # Step 2: Perform segmentation crop
        segmented_image, error_message = process_single_ktp_image(input_image_path)
        if error_message:
            return error_message

        # Step 3: Adjust image rotation
        rotated_image = rotate_if_necessary(segmented_image)

        # Step 4: Detect and crop fields
        cropped_images, error_message = detect_and_crop(rotated_image)
        if error_message:
            return error_message

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
