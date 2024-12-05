import os
import cv2
from .utils.cropField import detect_and_crop
from .utils.OCR import extractText
from .utils.orientation import detect_field_in_rotated_image
from .utils.segmentationCrop import process_single_ktp_image


def ktp_ocr(
    input_image_path: str,
    segmentation_model_output_folder: str,
    cropped_fields_folder: str,
):
    try:
        # Step 1: Perform orientation correction
        # print("[STEP 1] Orientation Detection and Correction")
        # print("-> Detecting fields and correcting orientation...")
        # oriented_image = detect_field_in_rotated_image(input_image_path)

        # if oriented_image is None:
        #     print("[ERROR] Orientation correction failed. Exiting.")
        #     return

        # oriented_image_path = os.path.join(segmentation_model_output_folder, "oriented_image.jpg")
        # cv2.imwrite(oriented_image_path, oriented_image)
        # print(f"[INFO] Oriented image saved to {oriented_image_path}\n")

        # Step 2: Perform segmentation crop
        segmented_image = process_single_ktp_image(input_image_path)

        if type(segmented_image) is str:
            print("[ERROR] Segmentation failed. Exiting.")
            return

        segmented_image_path = os.path.join(
            segmentation_model_output_folder, "segmented_image.jpg"
        )
        cv2.imwrite(segmented_image_path, segmented_image)

        # Step 3: Detect and crop fields
        cropped_images = detect_and_crop(
            segmented_image_path,
            cropped_fields_folder,
            confidence_threshold=0.5,
            save_crops=True,
        )

        # Step 4: Perform OCR on each cropped field
        if cropped_images:
            # for idx, cropped_image in tqdm(enumerate(cropped_images), total=len(cropped_images), desc="OCR Progress"):
            #     cropped_image_path = os.path.join(cropped_fields_folder, f"crop_{idx}.jpg")

            try:
                return extractText(cropped_images)
            except Exception as e:
                print(f"[ERROR] {e}")
        else:
            print("[INFO] No cropped fields to perform OCR on.\n")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
