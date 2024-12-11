# KTP Detection and Cropping 

## Overview
This project is a continuation of the machine learning process after carrying out the process of detecting KTP and non-KTP as well as the Segmentation and crop procces. The project contains the YOLOv8 model for labelling for box information(NIK,NAMA,and Alamat) and code to detect and crop regions from KTP images. The cropped regions can be used for further Model OCR processing. 

## Examples
### Detection and Cropping Results
Below are examples of detection and cropping performed by the model. The images on the left show the original KTP, where the model detects the information regions on the KTP with a bounding box around the detected areas and cropped regions. The images on the right show the cropped regions.

| **Original Image KTP**            | **Cropped Regions**                                              |
|-----------------------------------|------------------------------------------------------------------|
| ![Original Detection](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/Object_Detection_Bounding_Box/inputs-ktp/test-ktp.jpg) | ![Cropped Region 1](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/Object_Detection_Bounding_Box/output-crop/crop_0.jpg)<br>![Cropped Region 2](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/Object_Detection_Bounding_Box/output-crop/crop_1.jpg)<br>![Cropped Region 3](https://raw.githubusercontent.com/Capstone-Borwita/machine-learning-path/main/Object_Detection_Bounding_Box/output-crop/crop_2.jpg) |

## Folder Structure
- `inputs-ktp/`: Folder for input KTP images.
- `output-crop/`: Folder where cropped images will be saved.
- `Train result/`: Folder containing YOLOv8 model weights and training results.
- `crop_bounding_box.ipynb and detection-crop ktp.ipynb` : Jupyter notebook for local testing and execution.
- `detect_and_crop.py`: Python script for detecting and cropping regions.
- `requirements.txt`: List of Python dependencies.

