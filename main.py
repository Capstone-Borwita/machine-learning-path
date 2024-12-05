import os
import cv2
from tqdm import tqdm
from utils.cropField import detect_and_crop
from utils.OCR import extractText
from utils.orientation import detect_field_in_rotated_image
from utils.segmentationCrop import process_single_ktp_image

# Define paths
INPUT_IMAGE = r"testImages\testValidCrop\fotokopiKTP2.jpg"
SEGMENTATION_MODEL_OUTPUT_FOLDER = "output_images/segmentation"
CROPPED_FIELDS_FOLDER = "output_images/crops"

# Ensure output directories exist
os.makedirs(SEGMENTATION_MODEL_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FIELDS_FOLDER, exist_ok=True)

def main():
    try:
        print("\n[INFO] Starting KTP processing pipeline...\n")

        # Step 1: Perform orientation correction
        # print("[STEP 1] Orientation Detection and Correction")
        # print("-> Detecting fields and correcting orientation...")
        # oriented_image = detect_field_in_rotated_image(INPUT_IMAGE_PATH)

        # if oriented_image is None:
        #     print("[ERROR] Orientation correction failed. Exiting.")
        #     return

        # oriented_image_path = os.path.join(SEGMENTATION_MODEL_OUTPUT_FOLDER, "oriented_image.jpg")
        # cv2.imwrite(oriented_image_path, oriented_image)
        # print(f"[INFO] Oriented image saved to {oriented_image_path}\n")

        # Step 2: Perform segmentation crop
        print("[STEP 2] Segmentation")
        print("-> Segmenting the KTP from the oriented image...")
        segmented_image = process_single_ktp_image(INPUT_IMAGE)

        if type(segmented_image) is str:
            print("[ERROR] Segmentation failed. Exiting.")
            return

        segmented_image_path = os.path.join(SEGMENTATION_MODEL_OUTPUT_FOLDER, "segmented_image.jpg")
        cv2.imwrite(segmented_image_path, segmented_image)
        print(f"[INFO] Segmented image saved to {segmented_image_path}\n")

        # Step 3: Detect and crop fields
        print("[STEP 3] Field Detection and Cropping")
        print("-> Detecting fields in the segmented image...")
        cropped_images = detect_and_crop(
            segmented_image_path, CROPPED_FIELDS_FOLDER, confidence_threshold=0.5, save_crops=True
        )

        if not cropped_images:
            print("[WARNING] No fields detected in the image.\n")
        else:
            print(f"[INFO] Detected and saved {len(cropped_images)} cropped fields.\n")

        # Step 4: Perform OCR on each cropped field
        print("[STEP 4] OCR Processing")
        print("-> Performing OCR on cropped fields...")
        if cropped_images:
            # for idx, cropped_image in tqdm(enumerate(cropped_images), total=len(cropped_images), desc="OCR Progress"):
            #     cropped_image_path = os.path.join(CROPPED_FIELDS_FOLDER, f"crop_{idx}.jpg")

            try:
                results = extractText(cropped_images)
                for cls, text in results.items():
                    print(f"Extracted Text for Field {cls}: {text}")
            except Exception as e:
                print(f"[ERROR] {e}")
        else:
            print("[INFO] No cropped fields to perform OCR on.\n")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
