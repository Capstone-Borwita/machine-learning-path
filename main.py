import os
import cv2
from tqdm import tqdm
from utils.cropField import detect_and_crop
from utils.OCR import extractText
from utils.autoRotate import rotate_if_necessary
from utils.segmentationCrop import process_single_ktp_image
from utils.classification import classify_ktp

# Define paths
INPUT_FOLDER = r"testImages\mainTestImage" 
SEGMENTATION_MODEL_OUTPUT_FOLDER = r"testImages/output_images/segmentation"
CROPPED_FIELDS_FOLDER = r"testImages/output_images/crops"
RESULTS_FILE = r"testImages/output_images/ocr_results.txt"


# Ensure output directories exist
os.makedirs(SEGMENTATION_MODEL_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FIELDS_FOLDER, exist_ok=True)

def ktp_ocr(input_image_path: str):
    status = "Success"
    try:
        # Step 1: Classify KTP
        classification_result = classify_ktp(input_image_path)

        if not classification_result[0]:
            return "KTP tidak terdeteksi, harap coba lagi", "Failed at classification"

        # Step 2: Perform segmentation crop
        segmented_image, error_message = process_single_ktp_image(input_image_path)
        if error_message:
            return error_message, "Failed at segmentation"

        # Step 3: Adjust image rotation
        rotated_image = rotate_if_necessary(segmented_image)

        # Step 4: Detect and crop fields
        cropped_images, error_message = detect_and_crop(rotated_image)
        if error_message:
            return error_message, "Failed at field detection"

        # Step 5: Perform OCR on each cropped field
        if cropped_images:
            try:
                ocr_result = extractText(cropped_images)
                return ocr_result, status
            except Exception as e:
                return f"[ERROR] {e}", "Failed at OCR"
        else:
            return "No cropped fields to perform OCR on.", "Failed at cropping"

    except Exception as e:
        return f"[ERROR] An unexpected error occurred: {e}", "Unexpected failure"

def main():
    try:
        print("\n[INFO] Starting KTP processing pipeline...\n")

        # Create Result FIle
        with open(RESULTS_FILE, "w") as results_file:
            results_file.write("OCR Results\n")
            results_file.write("============\n")

        # Process each image in the input folder
        for image_file in tqdm(os.listdir(INPUT_FOLDER), desc="Processing images"):
            input_image_path = os.path.join(INPUT_FOLDER, image_file)

            if not os.path.isfile(input_image_path):
                continue

            print(f"\n[INFO] Processing image: {image_file}\n")

            # Run KTP OCR pipeline
            ocr_result, status = ktp_ocr(input_image_path)

            # Save results
            with open(RESULTS_FILE, "a") as results_file:
                results_file.write(f"Image: {image_file}\n")
                results_file.write(f"Status: {status}\n")

                if isinstance(ocr_result, dict):
                    for field, text in ocr_result.items():
                        results_file.write(f"  {field}: {text}\n")
                else:
                    results_file.write(f"  Message: {ocr_result}\n")

                results_file.write("\n")

            print(f"[INFO] Processing completed for {image_file} with status: {status}\n")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
