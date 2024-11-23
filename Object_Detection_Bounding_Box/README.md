# KTP Detection and Cropping Pipeline

## Overview
This project contains the YOLOv8 model and code to detect and crop regions from KTP images. The cropped regions can be used for further OCR processing.

## Folder Structure
- `inputs-ktp/`: Folder for input KTP images.
- `output-crop/`: Folder where cropped images will be saved.
- `Train result/`: Folder containing YOLOv8 model weights (`Model_Detection_Label.pt`) and training results.
- `crop_bounding_box.ipynb and detection-crop ktp.ipynb` : Jupyter notebook for local testing and execution.
- `detect_and_crop.py`: Python script for detecting and cropping regions.
- `requirements.txt`: List of Python dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
